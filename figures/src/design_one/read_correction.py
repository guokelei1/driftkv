"""Four stages of compact history encoding and read correction.

Publication schematic for one layer at the current release. No results are read
and no model is run. Original cache entries stay native; their narrow learned
codes are retained and their output projections are applied inside each read.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"figures/out/design_one"
INK,GRAY,NATIVE="#282B2E","#687079","#BDC4CA"
BLUE,ORANGE,RULE="#245D86","#BB610D","#D7DADD"


def text(ax,x,y,value,size=8,color=INK,ha="left",weight="normal"):
    return ax.text(x,y,value,fontsize=size,color=color,ha=ha,va="center",fontweight=weight)


def arrow(ax,start,end,color=GRAY,lw=.85):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=6,
        linewidth=lw,color=color,shrinkA=0,shrinkB=0))


def route(ax,points,color=GRAY,lw=.85):
    xs,ys=zip(*points[:-1])
    ax.plot(xs,ys,color=color,linewidth=lw,solid_capstyle="round")
    arrow(ax,points[-2],points[-1],color,lw)


def row(ax,x,y,w,h,color,cells=4):
    for i in range(cells):
        ax.add_patch(Rectangle((x+i*w/cells,y),w/cells,h,facecolor=color,
                              edgecolor="white",linewidth=.35))


def memory(ax,x,y,w,color,cells=4):
    for i in range(5): row(ax,x,y+i*10,w,7,color,cells)


def encoder(ax,x,y,w,h,color):
    # Contracting learned map, rather than a generic workflow box.
    verts=[(x,y),(x+w,y+h*.32),(x+w,y+h*.68),(x,y+h),(x,y)]
    ax.add_patch(PathPatch(MplPath(verts),facecolor="white",edgecolor=color,
                           linewidth=1.0,joinstyle="round"))
    for j in range(3):
        ax.add_patch(Circle((x+7,y+h*(.25+.25*j)),1.3,facecolor=color,edgecolor="none"))
    ax.plot([x+w*.39,x+w*.5,x+w*.65,x+w*.82],
            [y+h*.35,y+h*.35,y+h*.62,y+h*.65],color=color,linewidth=.9)


def plus(ax,x,y):
    ax.add_patch(Circle((x,y),8,facecolor="white",edgecolor=BLUE,linewidth=1.0))
    ax.plot([x-4,x+4],[y,y],color=BLUE,linewidth=1.0)
    ax.plot([x,x],[y-4,y+4],color=BLUE,linewidth=1.0)


def query(ax,x,y,label=True):
    ax.add_patch(Circle((x,y),8,facecolor="white",edgecolor=BLUE,linewidth=.9))
    text(ax,x,y,r"$q$",9,BLUE,ha="center")
    if label:
        text(ax,x,y+18,"Query",7.5,BLUE,ha="center")


def headings(ax):
    text(ax,159,261,"Prepare once / append",8.7,GRAY,ha="center",weight="bold")
    text(ax,512,261,"For each query",8.7,GRAY,ha="center",weight="bold")
    ax.plot([9,310],[249,249],color=RULE,lw=.7)
    ax.plot([331,693],[249,249],color=RULE,lw=.7)
    for number,x,title in [(1,16,"Encode history"),(2,174,"Reuse the codes"),
                           (3,337,"Correct attention"),(4,557,"Refine response")]:
        ax.add_patch(Circle((x,232),7.2,facecolor=INK,edgecolor="none"))
        text(ax,x,232,str(number),7.5,"white",ha="center",weight="bold")
        text(ax,x+12,232,title,8.4,weight="bold")
    ax.plot([321,321],[65,137],color=RULE,lw=.6)
    ax.plot([321,321],[171,217],color=RULE,lw=.6)


def prepare(ax):
    text(ax,35,179,"Native K/V",7.5,GRAY,ha="center")
    row(ax,12,152,32,7,NATIVE)
    row(ax,12,142,32,7,NATIVE)
    row(ax,12,119,32,7,NATIVE)
    text(ax,30,103,"Current",7.5,GRAY,ha="center")
    text(ax,30,90,r"item $e_i$",7.5,GRAY,ha="center")
    # A bracket denotes concatenation, not arithmetic addition.
    ax.plot([49,53,53,49],[158,158,119,119],color=GRAY,lw=.85)
    arrow(ax,(53,141),(65,141),GRAY)
    encoder(ax,69,121,42,40,ORANGE)
    text(ax,91,180,r"$f_\theta$",9,ORANGE,ha="center")
    text(ax,94,103,"Shared",7.5,ORANGE,ha="center")
    text(ax,94,90,"encoder",7.5,ORANGE,ha="center")
    arrow(ax,(115,141),(126,141),ORANGE)
    row(ax,130,136,11,10,ORANGE,cells=2)
    text(ax,135,166,r"$z_i$",8.5,ORANGE,ha="center")
    arrow(ax,(146,141),(172,141),ORANGE)
    # Stored codes are narrow and aligned with unchanged native cache rows.
    for x,w,c,cells,label in [(177,12,ORANGE,2,"Z"),(203,36,NATIVE,4,"K"),(251,36,NATIVE,4,"V")]:
        memory(ax,x,117,w,c,cells)
        text(ax,x+w/2,180,label,8,ORANGE if label=="Z" else GRAY,ha="center")
    text(ax,183,102,"Stored",7.5,ORANGE,ha="center")
    text(ax,183,89,"codes",7.5,ORANGE,ha="center")
    text(ax,249,102,"Native K/V",7.5,GRAY,ha="center")
    text(ax,249,89,"unchanged",7.5,GRAY,ha="center")
    arrow(ax,(294,141),(366,141),GRAY,lw=1.0)
    text(ax,330,155,"K/V + codes",7.5,GRAY,ha="center")


def attention(ax):
    # Key side: project q; never expand a full corrected key matrix.
    query(ax,346,194)
    arrow(ax,(356,194),(373,194),BLUE)
    text(ax,399,194,r"$P_K^{\top}q$",9,ORANGE,ha="center")
    text(ax,399,177,"Correct key scores",7.5,ORANGE,ha="center")
    arrow(ax,(399,168),(399,160),BLUE)
    # Weights vary in length only, without additional shades or opacity scales.
    for x,h in zip((374,382,390,398,406),(12,26,18,33,23)):
        ax.add_patch(Rectangle((x,126),4.7,h,facecolor=BLUE,edgecolor="none"))
    text(ax,392,111,"Corrected",7.5,BLUE,ha="center")
    text(ax,392,98,r"weights $\alpha_i$",7.5,BLUE,ha="center")
    arrow(ax,(416,141),(463,141),BLUE,lw=1.0)
    text(ax,441,158,"Read V",7.5,GRAY,ha="center")
    # Value side: project the weighted code sum after aggregation.
    text(ax,465,97,r"$P_V\!\left(\sum_i\alpha_i z_i\right)$",8.5,ORANGE,ha="center")
    text(ax,458,76,"Aggregate, then project",7.5,ORANGE,ha="center")
    route(ax,[(465,111),(465,119),(472,119),(472,132)],ORANGE)
    plus(ax,472,141)
    arrow(ax,(482,141),(509,141),BLUE,lw=1.0)
    row(ax,514,128,10,26,BLUE,cells=1)
    text(ax,521,108,r"$\tilde r$",8.5,BLUE,ha="center")
    text(ax,521,198,"Corrected",7.5,BLUE,ha="center")
    text(ax,521,185,"read",7.5,BLUE,ha="center")


def response(ax):
    # Both residual and bypass consume the actual corrected read.
    route(ax,[(530,141),(648,141)],BLUE,lw=1.0)
    route(ax,[(551,141),(551,178),(584,178)],BLUE)
    query(ax,564,197,label=False)
    arrow(ax,(571,192),(587,185),BLUE)
    encoder(ax,589,162,40,33,BLUE)
    text(ax,613,215,r"Residual model $g_\psi$",7.5,BLUE,ha="center")
    route(ax,[(634,178),(657,178),(657,150)],BLUE)
    text(ax,650,195,r"$\Delta r$",8.5,BLUE,ha="center")
    plus(ax,657,141)
    arrow(ax,(667,141),(681,141),BLUE,lw=1.0)
    row(ax,685,128,8,26,BLUE,cells=1)
    text(ax,681,110,r"$\hat r$",8.5,BLUE,ha="center")
    text(ax,668,92,"Final read",7.5,BLUE,ha="center")


def footer(ax):
    for x,label in [(80,"One code per entry"),(234,"Reuse across queries"),
                    (433,"Project within the read"),(623,"Query-specific residual")]:
        text(ax,x,52,label,7.5,ha="center",weight="bold")
    ax.plot([9,693],[34,34],color=RULE,lw=.6)
    text(ax,10,17,"Shared encoder, projections and response fitted on a few users against Full reads.",7.5,GRAY)


def main():
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans","Arial","Helvetica"],
        "mathtext.fontset":"dejavusans","font.size":8,"pdf.fonttype":42,
        "svg.fonttype":"none","savefig.facecolor":"white"})
    fig,ax=plt.subplots(figsize=(7,2.75))
    fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
    ax.set(xlim=(0,700),ylim=(0,275));ax.axis("off")
    headings(ax);prepare(ax);attention(ax);response(ax);footer(ax)
    OUT.mkdir(parents=True,exist_ok=True)
    for suffix in ("pdf","svg","png"):
        fig.savefig(OUT/f"read_correction.{suffix}",dpi=400)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=fig.bbox
    outside=[t.get_text() for t in ax.texts if not bounds.contains(*t.get_window_extent(renderer).get_points()[0])
             or not bounds.contains(*t.get_window_extent(renderer).get_points()[1])]
    print({"output":str(OUT),"size_inches":[7,2.75],"smallest_font_pt":min(t.get_fontsize() for t in ax.texts),
           "text_outside_canvas":outside})
    plt.close(fig)


if __name__=="__main__":
    main()
