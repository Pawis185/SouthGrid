"""Separate SVG guidance preview; training camera pixels are never touched."""
from html import escape
from .alignment import markers

def svg(frame, source):
    scene=markers(frame);box=scene['box']
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="540" viewBox="0 0 1000 540">',
         '<rect width="1000" height="540" fill="#102030"/>',
         '<g font-family="sans-serif" font-size="13" fill="white">',
         '<text x="20" y="25">Task3 host schematic (not camera projection) · '+escape(source)+'</text>']
    for panel,axis in enumerate((1,2)):
        cx=250+panel*500;cy=290;scale=230
        def point(p):return cx+(p[0]-box[0])*scale,cy-(p[axis]-box[axis])*scale
        out.append(f'<text x="{cx-200}" y="60">World X{["X","Y","Z"][axis]} / metres</text>')
        out.append(f'<circle cx="{cx}" cy="{cy}" r="{.3*scale}" fill="none" stroke="#617b91"/>')
        out.append(f'<path d="M {cx-8} {cy} h 16 M {cx} {cy-8} v 16" stroke="#74e4b7"/>')
        for tool,color in zip(scene['tools'],['#f8bc65','#67e1b5','#82a8ff','#ee8cbb','#c5a4ff']):
            for key,tag in [('current','current'),('grasp','grasp reference'),('release','release reference')]:
                x,y=point(tool[key]);out.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>')
                out.append(f'<text x="{x+5:.2f}" y="{y-5:.2f}" fill="{color}">{escape(tool["label"]+" "+tag)}</text>')
    out.extend(['<text x="20" y="520">Observed guidance only; full-score theoretical peak is at toolbox site. Raw images remain unmarked.</text>','</g></svg>'])
    return ''.join(out)
