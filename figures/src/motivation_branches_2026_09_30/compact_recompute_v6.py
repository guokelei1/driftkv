"""Compact selector mechanisms with explicit top-k score-based selection.

Draws four unequal-width selectors in one row on a 700×165 canvas.
Masks, discrepancy bars and attention weights are schematic, not measurements.
"""

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.transforms import Affine2D


INK = "#272727"
MUTED = "#68707A"
OLD = "#DEE3E9"
LINE = "#AAB2BC"
WARM = "#B64342"
WARM_CELL = "#E9A6A1"
BLUE = "#0F4D92"


def txt(ax,x,y,s,size=7,color=INK,weight="normal",ha="center",**kw):
    ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va="center",
            zorder=10,**kw)


def arr(ax,a,b,color=MUTED,lw=.7,head=4):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",color=color,lw=lw,
        mutation_scale=head,shrinkA=0,shrinkB=0,zorder=4))


def tiles(ax,x,y,w,h,rows=3,cols=6,selected=()):
    hits=set(selected)
    dx,dy=w/cols,h/rows
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle((x+c*dx+.4,y+r*dy+.4),dx-.8,dy-.8,
                fc=WARM_CELL if (r,c) in hits else OLD,
                ec="white",lw=.15,zorder=3))


def group(ax,x,y,w,h):
    ax.add_patch(FancyBboxPatch((x,y),w,h,
        boxstyle="round,pad=0,rounding_size=3",fc="#FCFCFD",
        ec="#D8DDE3",lw=.55,zorder=0))


def draw_recompute(ax):
    """Shared rebuild strip above Layer, Tail, Deviation, and Query selectors."""
    txt(ax,228,155,"(a) Selective recomputation",8.7,weight="bold")
    previous_artists = set(ax.get_children())
    tiles(ax,12,126,62,9,rows=1,cols=6)
    for c in (1,4):
        ax.add_patch(Rectangle((12+c*62/6,125.5),62/6,10,
            fc="none",ec=WARM,lw=.7,linestyle=(0,(2,1.2)),zorder=5))
    arr(ax,(83,130.5),(142,130.5),WARM,.75)
    txt(ax,112,141,"Recompute",7.0,WARM,"bold")
    txt(ax,112,119,"current model",6.3,MUTED)
    tiles(ax,151,126,62,9,rows=1,cols=6,
          selected={(0,1),(0,4)})
    ax.add_patch(Rectangle((244,131),6,6,fc=WARM_CELL,ec="none"))
    txt(ax,257,134,"selected K/V rebuilt",6.3,MUTED,ha="left")
    ax.add_patch(Rectangle((244,117),6,6,fc=OLD,ec="none"))
    txt(ax,257,120,"other K/V reused",6.3,MUTED,ha="left")
    # Center the complete shared operation, including its legend, rather than
    # centering just the arrow while the rest drifts to one side.
    shared = [a for a in ax.get_children() if a not in previous_artists]
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    bounds = [a.get_window_extent(renderer) for a in shared]
    actual_center = (min(b.x0 for b in bounds) + max(b.x1 for b in bounds)) / 2
    target_center = ax.transData.transform((228, 0))[0]
    shift = (
        ax.transData.inverted().transform((target_center, 0))[0]
        - ax.transData.inverted().transform((actual_center, 0))[0])
    for artist in shared:
        artist.set_transform(Affine2D().translate(shift, 0) + ax.transData)

    # Simple policies take less width; the two scoring policies need room for
    # their distinct score sources. Cards share a single horizontal baseline.
    for x,w in ((8,70),(86,70),(164,134),(306,142)):
        group(ax,x,9,w,98)

    txt(ax,43,97,"Layer",7.7,WARM,"bold")
    txt(ax,43,83,"Full-logit error",6.2,INK)
    for c,h in enumerate((13,4,10)):
        ax.add_patch(Rectangle((23+c*10,58),6,h,
            fc=WARM if c==1 else LINE,ec="none",zorder=3))
    arr(ax,(59,68),(59,53),WARM,.7,3.5)
    tiles(ax,19,29,48,24,rows=4,cols=4,
          selected={(r,c) for r in (1,2) for c in range(4)})
    txt(ax,43,18,"Best span",6.7,WARM,"bold")

    txt(ax,121,97,"Tail",7.7,WARM,"bold")
    txt(ax,121,83,"Event time",6.2,INK)
    tiles(ax,94,44,54,29,rows=3,cols=6,
          selected={(r,c) for r in range(3) for c in (4,5)})
    arr(ax,(94,35),(148,35),MUTED,.65,3.5)
    txt(ax,94,29,"past",6.0,MUTED,ha="left")
    txt(ax,148,29,"now",6.0,MUTED,ha="right")
    txt(ax,121,18,"Latest k",6.8,WARM,"bold")

    # Compare corresponding old/current first-layer K/V, then rank the
    # resulting scores. Input colors identify producers, not selected rows.
    txt(ax,231,97,"Deviation-guided",7.5,WARM,"bold")
    txt(ax,231,83,"Compare first-layer K/V",6.3,INK)
    txt(ax,171,69,"old",6.3,INK,ha="left")
    txt(ax,171,55,"new",6.3,BLUE,ha="left")
    deviation_scores=(.20,.90,.25,.10,.70)
    deviation_topk=set(sorted(range(len(deviation_scores)),
                              key=lambda i: deviation_scores[i])[-2:])
    for c,score in enumerate(deviation_scores):
        x=202+c*17
        hot=c in deviation_topk
        h=12*score
        tiles(ax,x,66,11,6,rows=1,cols=2)
        for j in range(2):
            ax.add_patch(Rectangle((x+j*5.5+.4,52.4),4.7,5.2,
                fc="#B7CEE6",ec="white",lw=.15,zorder=3))
        txt(ax,x+5.5,62,"−",6.1,INK)
        arr(ax,(x+5.5,50),(x+5.5,45),MUTED,.55,2.8)
        ax.add_patch(Rectangle((x+3.5,32),4,h,
            fc=WARM if hot else LINE,ec="none",zorder=3))
    txt(ax,183,37,"scores",6.2,INK)
    txt(ax,231,18,"Select top-k positions",6.6,WARM,"bold")

    # The query and inherited first-layer keys jointly produce attention
    # scores. Input keys are unselected; only the largest output scores are
    # highlighted, so the graphic follows score -> select in causal order.
    txt(ax,377,97,"Query-guided",7.5,WARM,"bold")
    txt(ax,334,80,r"Query $q$",7.2,BLUE)
    txt(ax,400,86,"Inherited K",6.1,INK)
    for c in range(5):
        tiles(ax,364+c*15,73,10,7,rows=1,cols=1)
    arr(ax,(338,74),(354,66),BLUE,.85,3.8)
    arr(ax,(399,72),(399,66),INK,.85,3.8)
    ax.add_patch(FancyBboxPatch((324,49),110,17,
        boxstyle="round,pad=0,rounding_size=2",fc="#EDF4FC",
        ec="#ADC3DE",lw=.6,zorder=3))
    txt(ax,379,57.5,"Layer-1 attention",6.6,BLUE,"bold")
    arr(ax,(379,49),(379,44),BLUE,.85,3.8)
    txt(ax,329,36,"scores",6.2,INK)
    attention_scores=(.16,.29,.95,.12,.76)
    attention_topk=set(sorted(range(len(attention_scores)),
                              key=lambda i: attention_scores[i])[-2:])
    for c,score in enumerate(attention_scores):
        x=350+c*18
        hot=c in attention_topk
        ax.add_patch(Rectangle((x+2.5,30),5,12*score,
            fc=WARM if hot else LINE,ec="none",zorder=3))
    txt(ax,377,18,"Select top-k positions",6.6,WARM,"bold")
