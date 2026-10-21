"""Assemble already rendered SYNTHETIC figures; never invoke LaTeX or training."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pymupdf as fitz
from render_all import FIGURES, ROOT

FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
TITLES = ['整体质量与计算成本', '相同资源下的完整时间分解', '默认点跨发布的质量与成本',
          'Design：读取结构与校准', 'Implementation：准备与读取优化',
          '固定版本与读取次数的适用边界']
ORDER = [0, 2, 3, 4, 1, 5]


def contact_sheet():
    width, tile_w, gap, header = 2400, 1156, 28, 150
    tile_h, rows = 850, 3
    out = Image.new('RGB', (width, header + rows * (tile_h + gap) + 30), '#F2F4F6')
    draw = ImageDraw.Draw(out)
    draw.text((30, 16), 'CacheTide 实验章节 · 六幅实验图总览（模拟数据）',
              font=ImageFont.truetype(FONT, 38), fill='#22364B')
    draw.text((32, 83), 'Yambda / RecFlow × Large / Max；双栏主结果与单栏图表预览。所有数值均为模拟。',
              font=ImageFont.truetype(FONT, 23), fill='#626D77')
    for slot, i in enumerate(ORDER):
        folder, number, _ = FIGURES[i]
        title = TITLES[i]
        x, y = gap + (slot % 2) * (tile_w + gap), header + (slot // 2) * (tile_h + gap)
        draw.rounded_rectangle((x, y, x+tile_w, y+tile_h), radius=8, fill='white')
        code = FIGURES[i][0].split('_')[0].upper()
        draw.text((x+18, y+12), f'{code}  {title} · 模拟',
                  font=ImageFont.truetype(FONT, 23), fill='#263D51')
        image = Image.open(ROOT / folder / f'fig_{number}_simulated.png').convert('RGB')
        image.thumbnail((tile_w-18, tile_h-66), Image.Resampling.LANCZOS)
        out.paste(image, (x+(tile_w-image.width)//2, y+55+(tile_h-66-image.height)//2))
    out.save(ROOT / 'overview_simulated.png')


def paper_preview():
    """Render both core results wide and all mechanism/cost details narrow."""
    doc = fitz.open()
    # E1 spans both columns. Keep the two complementary E3 studies adjacent.
    pages = [[(0,), (2,)], [(3, 4), (1, 5)]]
    for page_no, rows in enumerate(pages, 1):
        page = doc.new_page(width=595.3, height=841.9)
        page.insert_text((35,30), 'CacheTide | double-column main result and column-width details', fontsize=11, fontname='hebo', color=(.12,.23,.32))
        page.insert_text((35,47), 'SIMULATED DATA - LAYOUT ONLY; NOT EXPERIMENTAL EVIDENCE', fontsize=8, color=(.63,.28,.16))
        row_y = 65
        for row in rows:
            row_height = 0
            for slot, index in enumerate(row):
                folder, number, caption = FIGURES[index]
                src = fitz.open(ROOT / folder / f'fig_{number}_simulated.pdf')
                x = 42 + slot * 270
                width = 511.2 if index in (0, 2) else 241.2
                height = width * src[0].rect.height / src[0].rect.width
                row_height = max(row_height, height)
                if row_y + height > 795:
                    raise ValueError(f'Asset too tall for preview page: {folder}')
                page.show_pdf_page(fitz.Rect(x,row_y,x+width,row_y+height),src,0)
                code = folder.split('_')[0].upper()
                page.insert_textbox(fitz.Rect(x,row_y+height+7,x+width,row_y+height+38), f'{code} (simulated). {caption}', fontsize=7.4,fontname='heit',color=(.2,.24,.28))
                src.close()
            row_y += row_height + 35
        page.insert_text((35,825),'Figure sizes match the intended column span. Manuscript: paper/main.pdf.',fontsize=7.5,color=(.45,.48,.51))
    doc.save(ROOT/'paper_preview_simulated.pdf',garbage=4,deflate=True)
    for i,page in enumerate(doc):
        page.get_pixmap(matrix=fitz.Matrix(1.6,1.6)).save(ROOT/f'paper_preview_page_{i+1}_simulated.png')
    doc.close()
    for old in ROOT.glob('paper_preview_page_*_simulated.png'):
        if int(old.name.split('_')[3]) > len(pages): old.unlink()


if __name__ == '__main__':
    contact_sheet()
    paper_preview()
    print(ROOT / 'overview_simulated.png')
    print(ROOT / 'paper_preview_simulated.pdf')
