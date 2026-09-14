"""Crash-recoverable JSONL journal. Producer never waits on disk or scoring."""
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time
import uuid
import numpy as np


def json_default(v):
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, np.generic): return v.item()
    raise TypeError(type(v).__name__)

def dumps(v):
    return json.dumps(v, default=json_default, ensure_ascii=False, allow_nan=False, separators=(',', ':'))

def atomic(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(dumps(data), encoding='utf-8')
    os.replace(tmp, path)

def read_records(path):
    """Ignore only a torn final line; never skip malformed interior evidence."""
    result, torn = [], False
    with Path(path).open('rb') as f:
        for line in f:
            if not line.endswith(b'\n'):
                torn = True
                break
            result.append(json.loads(line))
    return result, torn

def _sink(path, q):
    path = Path(path)
    digest = hashlib.sha256()
    last_sync = last_preview = 0.
    last_physics = None
    with (path / 'raw.jsonl').open('xb', buffering=0) as f:
        while True:
            item = q.get()
            if item is None: break
            line = (dumps(item) + '\n').encode()
            f.write(line)
            digest.update(line)
            now = time.monotonic()
            if now - last_sync > .5:
                os.fsync(f.fileno()); last_sync = now
            if item['kind'] in {'physics','idle_preview'}: last_physics = item
            if item['kind'] in {'physics','idle_preview'} and now - last_preview > .2:
                atomic(path / 'latest.json', item); last_preview = now
        os.fsync(f.fileno())
    if last_physics is not None: atomic(path / 'latest.json', last_physics)
    atomic(path / 'raw_digest.json', {'sha256': digest.hexdigest(), 'bytes': (path/'raw.jsonl').stat().st_size})

class Journal:
    def __init__(self, root, metadata):
        self.id = uuid.uuid4().hex
        self.path = Path(root) / 'episodes' / self.id
        self.path.mkdir(parents=True)
        self.manifest = dict(schema_version=1, episode_uuid=self.id, state='open', metadata=metadata,
                             source='synthetic_test' if metadata.get('test_data') else 'real_orca_local_physics', official_attempts_sent=0,
                             scoring_complete=False, camera_exposure_timestamps_available=False)
        atomic(self.path / 'manifest.json', self.manifest)
        self.dropped = 0
        self.ctx = mp.get_context('spawn')
        self.q = self.ctx.Queue(maxsize=2048)
        self.process = self.ctx.Process(target=_sink, args=(str(self.path), self.q), daemon=True)
        self.process.start()
        self.seq = 0
        self.closed = False

    def emit(self, kind, **fields):
        item = dict(kind=kind, record_id=self.seq, host_wall_ns=time.time_ns(),
                    host_mono_ns=time.monotonic_ns())
        item.update(fields)
        self.seq += 1
        try:
            self.q.put_nowait(item)
        except queue.Full:
            self.dropped += 1
        if not self.process.is_alive(): self.dropped += 1
        return item

    def close(self, outcome, **fields):
        if self.closed: return
        self.emit('episode_end', outcome=outcome)
        try:
            self.q.put(None, timeout=5)
            self.process.join(10)
        except queue.Full:
            pass
        drained = not self.process.is_alive() and self.process.exitcode == 0
        self.manifest.update(state=outcome, dropped_records=self.dropped, journal_drained=drained,
                             scoring_complete=drained and not self.dropped and outcome in {"saved", "deleted", "discarded"},
                             last_record_id=self.seq - 1, **fields)
        atomic(self.path / 'manifest.json', self.manifest)
        self.closed = True
        # Do not block interpreter shutdown on a failed consumer.
        self.q.cancel_join_thread()
        self.q.close()
