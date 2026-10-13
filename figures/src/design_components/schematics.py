"""Mechanism architecture: a learned read view over one maintained native cache.

All matrices, weights and token rows are schematic. The generator reads no
experimental data and changes no method, model or result.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "figures/out/design_components"
INK, GRAY, RULE = "#282B2E", "#687079", "#D7DADD"
BLUE, ORANGE, PURPLE, TEAL = "#245D86", "#BB610D", "#77589C", "#327F78"
NATIVE, PALE_K, PALE_V = "#BDC4CA", "#C9CFD4", "#AABAC7"
PALE_Z = "#EDD5BD"


def text(ax,x,y,value,size=9,color=INK,ha="center",weight="normal",**kwargs):
    return ax.text(x,y,value,fontsize=size,color=color,ha=ha,va="center",
                   fontweight=weight,zorder=6,**kwargs)


def arrow(ax,start,end,color=BLUE,lw=1.2,dashed=False,head=7):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=head,
        linewidth=lw,color=color,shrinkA=0,shrinkB=0,
        linestyle=(0,(3,2)) if dashed else "-",zorder=5))


def route(ax,points,color=BLUE,lw=1.2,dashed=False,head=7):
    xs,ys=zip(*points[:-1])
    ax.plot(xs,ys,color=color,linewidth=lw,solid_capstyle="butt",
            linestyle=(0,(3,2)) if dashed else "-",zorder=5)
    arrow(ax,points[-2],points[-1],color,lw,dashed,head)


def curve(ax,points,color=BLUE,lw=1.1,alpha=1):
    ax.add_patch(PathPatch(MplPath(points,[MplPath.MOVETO,MplPath.CURVE4,
        MplPath.CURVE4,MplPath.CURVE4]),facecolor="none",edgecolor=color,
        linewidth=lw,alpha=alpha,zorder=3))


def vector(ax,x,y,w,h=12,n=5,color=BLUE,alpha=1,outline=False):
    cell=w/n
    for j in range(n):
        ax.add_patch(Rectangle((x+j*cell,y),cell-1.1,h,
            facecolor="white" if outline else color,
            edgecolor=color if outline else "white",linewidth=.55,
            alpha=alpha,zorder=4))


def funnel(ax,x,y,w,h,color,expand=False,label=None):
    small=h*.34
    if expand:
        points=[(x,y-small/2),(x+w,y-h/2),(x+w,y+h/2),(x,y+small/2)]
    else:
        points=[(x,y-h/2),(x+w,y-small/2),(x+w,y+small/2),(x,y+h/2)]
    ax.add_patch(Polygon(points,facecolor="white",edgecolor=color,lw=1.15,zorder=4))
    # Connection strokes denote a map between dimensions, not network depth.
    for f in (-.3,0,.3):
        left=y+f*(small if expand else h)
        right=y+f*(h if expand else small)
        ax.plot([x+4,x+w-4],[left,right],color=color,lw=.6,alpha=.7,zorder=4)
    if label:
        text(ax,x+w/2,y+h/2+12,label,10,color)


def badge(ax,x,y,n,label,color):
    ax.add_patch(Circle((x,y),7.8,facecolor=color,edgecolor="none",zorder=6))
    text(ax,x,y,str(n),9,"white",weight="bold")
    text(ax,x+14,y,label,10.5,color,ha="left",weight="bold")


def slab(ax,x,y,w,h,color=GRAY,fill="#E8EBEE"):
    ax.add_patch(Polygon([(x,y),(x+w,y),(x+w+5,y+5),(x+5,y+5)],
        facecolor=fill,edgecolor=color,lw=.7,zorder=3))
    ax.add_patch(Rectangle((x,y-h),w,h,facecolor=fill,edgecolor=color,lw=.7,zorder=3))
    for j in range(3):
        ax.plot([x+8,x+w-8],[y-h+4+j*3,y-h+4+j*3],color=color,lw=.6,alpha=.45,zorder=4)


def system_architecture():
    fig,ax=plt.subplots(figsize=(7.7,5.4))
    fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
    ax.set(xlim=(0,800),ylim=(0,560));ax.axis("off")

    # Offline preparation is compact and visually separate from live queries.
    ax.add_patch(Rectangle((10,455),780,95,facecolor="#FAF8FC",edgecolor="none",zorder=0))
    badge(ax,25,541,1,"Offline calibration",PURPLE)
    for yy in (505,491):
        for j in range(5):
            ax.add_patch(Circle((28+j*10,yy),2.5,facecolor=NATIVE,edgecolor="none",zorder=4))
        ax.plot([25,71],[yy,yy],color=RULE,lw=.6,zorder=2)
    text(ax,51,473,"Sample",9,GRAY)
    arrow(ax,(78,499),(103,499),GRAY)
    for y in (513,500,487):
        slab(ax,111,y,43,8)
    text(ax,134,467,"Current replay",9,GRAY)
    arrow(ax,(160,498),(177,498),GRAY)
    vector(ax,181,489,18,16,3,BLUE)
    vector(ax,210,483,15,22,4,PURPLE)
    text(ax,189,517,r"$r^\star$",10,BLUE)
    text(ax,218,517,r"$s^\star$",10,PURPLE)
    ax.plot([199,230],[497,497],color=BLUE,lw=.8)
    ax.plot([225,230],[494,494],color=PURPLE,lw=.8)
    ax.plot([230,230],[490,500],color=PURPLE,lw=.8)
    route(ax,[(230,494),(264,494),(264,470),(286,470)],PURPLE)

    # The three shared parameter objects are actual encoder/projection maps.
    funnel(ax,303,492,38,26,ORANGE,label=r"$f_\theta$")
    funnel(ax,385,492,35,25,ORANGE,label=r"$P_K$")
    funnel(ax,463,492,35,25,ORANGE,expand=True,label=r"$P_V$")
    ax.plot([286,514],[470,470],color=PURPLE,lw=1.0)
    for x in (321,402,480):
        arrow(ax,(x,470),(x,477),PURPLE,head=5)
    text(ax,406,454,"Local reads → joint scores",9,PURPLE)
    route(ax,[(514,470),(530,470),(530,487),(549,487)],PURPLE)
    ax.add_patch(Circle((570,487),15,facecolor="#F2EFF7",edgecolor=PURPLE,lw=1))
    text(ax,570,487,r"$\theta_v$",12,PURPLE)
    arrow(ax,(570,472),(570,416),PURPLE,dashed=True)
    ax.add_patch(Circle((570,413),3,facecolor=PURPLE,edgecolor="none",zorder=5))
    text(ax,603,449,"publish",9,PURPLE)
    text(ax,691,497,"Learned once",9,PURPLE)
    text(ax,691,477,"Shared across users",9,PURPLE)
    ax.plot([13,785],[445,445],color=RULE,lw=.8,linestyle=(0,(4,3)))

    # The visual center is one per-layer slice of one physical native cache.
    badge(ax,25,427,2,"Read through compact codes",BLUE)
    text(ax,628,423,r"fixed $\theta_v$",9,PURPLE)
    for shift in (16,8):
        ax.add_patch(Rectangle((228+shift,187+shift),231,213,
            facecolor="#F4F6F8",edgecolor="#AEB6BE",lw=.75,zorder=1))
    ax.add_patch(Rectangle((228,187),231,213,facecolor="white",edgecolor=GRAY,lw=1,zorder=2))
    text(ax,256,382,"K",11,GRAY,weight="bold")
    text(ax,333,382,"V",11,GRAY,weight="bold")
    text(ax,391,382,"Z",11,ORANGE,weight="bold")
    text(ax,432,382,"c",10,TEAL,weight="bold")
    row_y=(348,321,294,267,240,213)
    support=(1,1,1,1,0,0)
    for y,c in zip(row_y,support):
        vector(ax,237,y,61,15,4,PALE_K)
        vector(ax,311,y,61,15,4,PALE_V)
        vector(ax,383,y,17,15,2,ORANGE if c else PALE_Z,outline=not c)
        if not c:
            text(ax,391.5,y+7.5,"0",9,TEAL)
        text(ax,432,y+7.5,str(c),9,TEAL)
    text(ax,344,198,"One native cache + codes",9,GRAY)

    # Materializing a row code happens on first read and append, away from q.
    vector(ax,26,315,53,10,5,PALE_K)
    vector(ax,26,294,53,10,5,PALE_V)
    vector(ax,26,265,53,10,5,BLUE)
    text(ax,17,320,r"$k_i$",9,GRAY,ha="right")
    text(ax,17,299,r"$v_i$",9,GRAY,ha="right")
    text(ax,17,270,r"$e_i$",9,BLUE,ha="right")
    route(ax,[(82,320),(95,320),(106,309)],GRAY)
    arrow(ax,(82,299),(105,299),GRAY)
    route(ax,[(82,270),(95,270),(106,290)],BLUE)
    funnel(ax,107,301,43,43,ORANGE,label=r"$f_\theta$")
    arrow(ax,(151,301),(168,301),ORANGE)
    vector(ax,169,295,20,12,3,ORANGE)
    route(ax,[(193,301),(207,301),(207,415),(384,415)],ORANGE)
    ax.add_patch(Circle((391,415),7,facecolor="white",edgecolor=TEAL,lw=1,zorder=6))
    text(ax,391,415,"×",10,TEAL)
    route(ax,[(432,400),(432,415),(398,415)],TEAL,head=5)
    arrow(ax,(391,408),(391,397),ORANGE,head=5)
    text(ax,137,248,"First read / append",9,GRAY)
    text(ax,106,227,"Current item + native row",9,GRAY)

    # An original query and a query projected into code space determine row
    # weights. A pair fan depicts the SAME weights reading native V and Z.
    vector(ax,581,365,57,13,6,BLUE)
    text(ax,609,395,r"$q_\ell$",11,BLUE)
    # A query flows right to left: the wide side is its input.
    funnel(ax,525,371,40,29,ORANGE,expand=True,label=r"$P_K$")
    arrow(ax,(578,371),(567,371),BLUE,head=5)
    vector(ax,489,366,21,11,3,ORANGE)
    arrow(ax,(523,371),(514,371),ORANGE,head=5)
    route(ax,[(610,365),(610,354),(520,354),(514,354)],BLUE,head=5)
    arrow(ax,(499,364),(499,361),ORANGE,head=5)
    text(ax,500,192,r"$\alpha(q;K,Z)$",10,BLUE)
    widths=(13,22,8,17,10,15)
    for y,w in zip(row_y,widths):
        arrow(ax,(463,y+7.5),(486,y+7.5),BLUE,lw=.65,head=4)
        ax.add_patch(Rectangle((489,y+3),24,9,facecolor="white",edgecolor=RULE,lw=.45,zorder=3))
        vector(ax,489,y+3,w,9,1,BLUE)
    # Three representative row streams suffice; row line widths are schematic.
    for y,lw in [(355.5,1.0),(301.5,1.6),(247.5,.9)]:
        ax.add_patch(Circle((515,y),1.35,facecolor=BLUE,edgecolor="none",zorder=5))
        curve(ax,[(515,y),(541,y),(534,305),(548,305)],BLUE,lw=lw,alpha=.85)
        curve(ax,[(515,y),(552,y),(535,263),(548,263)],ORANGE,lw=lw,alpha=.8)
    for yy,color in ((305,BLUE),(263,ORANGE)):
        ax.add_patch(Circle((552,yy),5,facecolor="white",edgecolor=color,lw=.8,zorder=5))
        text(ax,552,yy,r"$\Sigma$",9,color)
        arrow(ax,(558,yy),(562,yy),color,head=4)
    vector(ax,563,299,46,13,5,BLUE)
    vector(ax,563,257,20,12,3,ORANGE)
    text(ax,585,324,r"$\bar v$",11,BLUE)
    text(ax,573,242,r"$\bar z$",11,ORANGE)
    # A brace associates weight generation and both accumulators with ONE pass.
    ax.plot([477,477,613,613],[215,206,206,215],color=BLUE,lw=.8)
    text(ax,578,172,"Same weights · one scan",9,BLUE,weight="bold")
    arrow(ax,(587,263),(600,263),ORANGE,head=5)
    funnel(ax,602,263,33,30,ORANGE,expand=True,label=r"$P_V$")
    arrow(ax,(637,263),(644,263),ORANGE,head=5)
    vector(ax,646,257,46,13,5,ORANGE)
    text(ax,671,242,"Decode once",9,ORANGE)
    route(ax,[(611,305),(705,305),(705,287)],BLUE,lw=1.5)
    route(ax,[(694,263),(705,263),(705,271)],ORANGE,lw=1.4)
    ax.add_patch(Circle((705,279),8,facecolor="white",edgecolor=BLUE,lw=1.1,zorder=6))
    text(ax,705,279,"+",12,BLUE)
    arrow(ax,(714,279),(730,279),BLUE,lw=1.5)
    vector(ax,732,273,46,13,5,BLUE)
    text(ax,755,301,r"$\tilde r_\ell$",11,BLUE)

    # The opened Current layer shows where the maintained read view is used.
    text(ax,742,410,"Current model",9,GRAY,weight="bold")
    for y in (386,229,192):
        slab(ax,730,y,48,17)
    route(ax,[(729,377),(668,377),(668,371),(642,371)],BLUE)
    route(ax,[(780,279),(786,279),(786,242),(754,242),(754,231)],BLUE)
    arrow(ax,(754,210),(754,197),BLUE)
    arrow(ax,(754,174),(754,148),BLUE)
    for y,w in [(133,27),(121,39),(109,18)]:
        vector(ax,734,y,w,6,1,BLUE)
    text(ax,757,92,"Scores",9,BLUE)
    ax.plot([721,716,716,721],[356,356,318,318],color=GRAY,lw=.8)
    text(ax,758,344,r"Layer $\ell$",9,GRAY)

    # State evolution uses real objects: an arriving row is appended as one
    # aligned KV/Z/c record, while eviction removes that same complete record.
    badge(ax,25,155,3,"Maintain aligned state",TEAL)
    ax.add_patch(Circle((27,114),6,facecolor=PALE_V,edgecolor=GRAY,lw=.7))
    text(ax,28,93,"Event",9,GRAY)
    arrow(ax,(37,114),(66,114),GRAY)
    for y in (126,113,100):
        slab(ax,74,y,38,8)
    text(ax,94,78,"Native writes",9,GRAY)
    arrow(ax,(119,113),(153,113),GRAY)
    vector(ax,157,106,60,14,6,PALE_V)
    arrow(ax,(221,113),(237,113),GRAY)
    # This is the entering record, not another cache or an expanded correction.
    for x,w,color,n in [(240,61,PALE_K,4),(314,61,PALE_V,4),(386,17,ORANGE,2)]:
        vector(ax,x,121,w,15,n,color)
    text(ax,433,128,r"$c_i$",9,TEAL)
    ax.add_patch(Rectangle((235,116),222,26,facecolor="none",edgecolor=TEAL,lw=.8,zorder=3))
    arrow(ax,(344,144),(344,184),TEAL,lw=1.25)
    text(ax,345,99,"Append together",9,TEAL)
    # Lower-layer ancestry may survive in Current-produced entering rows.
    for x in (383,409,435):
        ax.add_patch(Circle((x,55),3.2,facecolor=PALE_K,edgecolor=TEAL,lw=.7))
    curve(ax,[(383,59),(383,77),(432,79),(432,111)],TEAL,lw=1)
    arrow(ax,(432,104),(432,116),TEAL,head=5)
    text(ax,421,35,"Lower-layer ancestry",9,TEAL)
    ax.add_patch(Circle((228,356),6.5,facecolor="white",edgecolor=TEAL,lw=.8,zorder=6))
    text(ax,228,356,"−",10,TEAL)
    arrow(ax,(221,356),(214,356),TEAL,head=5)
    text(ax,166,365,"Evict row",9,TEAL)

    # Retirement is a structural state transition, represented by removing
    # only the code sidecar. It is not a learned online-quality decision.
    text(ax,609,129,"All-layer clear / rebuild",9,TEAL)
    vector(ax,527,81,37,14,4,ORANGE,alpha=.3,outline=True)
    ax.plot([528,563],[78,99],color=TEAL,lw=1.0,zorder=5)
    arrow(ax,(572,88),(601,88),TEAL)
    vector(ax,609,80,64,15,6,PALE_V)
    arrow(ax,(676,88),(704,88),BLUE)
    text(ax,634,61,"Native read",9,GRAY)
    return fig,ax


CAPTION=(
    "Integrated reading-view architecture. Only one native serving cache is kept; "
    "stacked outlines denote its layers, not corrected copies. Offline, sampled "
    "histories reconstructed by Current supply read and final-score targets for "
    "local initialization and joint alignment of the shared encoder and Key/Value "
    "projections. A release publishes those parameters. At first read and appends, "
    "head-local native K/V and the complete Current item embedding form row codes, gated by "
    "existing inherited-support flags. Codes are stored with their native rows. "
    "An incoming query and its Key projection select history weights from native "
    "K and codes. One traversal aggregates native values and the same codes with "
    "the same weights. The narrow aggregate is expanded once by the Value "
    "projection and added to the native aggregate inside the Current query path. "
    "Native event writes remain unchanged. Aligned appends and evictions maintain "
    "the KV/code/flag record; propagated lower-layer ancestry scopes the codes. "
    "All-layer clearance or an explicit reconstruction removes the codes and "
    "restores native reads. Clearing ancestry does not imply equality to Fresh. "
    "All glyph dimensions, weights and token examples are schematic; no measured values."
)


def main():
    plt.rcParams.update({
        "font.family":"sans-serif","font.sans-serif":["DejaVu Sans","Arial","Helvetica"],
        "mathtext.fontset":"dejavusans","font.size":9,"pdf.fonttype":42,
        "svg.fonttype":"none","savefig.facecolor":"white",
    })
    OUT.mkdir(parents=True,exist_ok=True)
    fig,ax=system_architecture()
    for suffix in ("pdf","svg","png"):
        fig.savefig(OUT/f"system_architecture.{suffix}",dpi=400)
    fig.canvas.draw();renderer,bounds=fig.canvas.get_renderer(),fig.bbox
    outside=[t.get_text() for t in ax.texts
        if any(not bounds.contains(*p) for p in t.get_window_extent(renderer).get_points())]
    sources=["figures/src/design_components/schematics.py",
        "src/hstu_kvcache/design_one/lightweight_fullitem.py",
        "src/hstu_kvcache/design_one/lightweight_fullitem_kv.py",
        "src/hstu_kvcache/design_one/lightweight_kv_dispatch.py",
        "src/hstu_kvcache/design_two/joint_local_kv_refinement.py",
        "scripts/design_system/evaluate_lightweight_support_mask.py",
        "src/hstu_kvcache/design_three/lifecycle.py"]
    spec={"size_inches":list(fig.get_size_inches()),"caption":CAPTION,
        "smallest_font_pt":min(t.get_fontsize() for t in ax.texts),
        "text_outside_canvas":outside,
        "outputs":[f"figures/out/design_components/system_architecture.{s}" for s in ("pdf","svg","png")]}
    manifest={"kind":"mechanism schematic; no measured values",
        "primary_figure":"system_architecture","figures":{"system_architecture":spec},
        "sources":[{"path":p,"sha256":hashlib.sha256((ROOT/p).read_bytes()).hexdigest()} for p in sources]}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    plt.close(fig)
    print(json.dumps({k:spec[k] for k in ("outputs","smallest_font_pt","text_outside_canvas")}))


if __name__=="__main__":
    main()
