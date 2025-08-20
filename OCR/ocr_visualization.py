# ocr_visualization.py
# -*- coding: utf-8 -*-
import json, cv2, numpy as np, os, math
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

# ============= 全局常量 =============
TEXT_W, TEXT_H   = 800, 200                # 进一步增大槽位高度以适应更大的中文字体
SECTORS          = ['E','NE','N','NW','W','SW','S','SE']
BASE_SCALE       = 3.0                     # 字体大小
ARROW_THICKNESS  = 2
ARROW_TIPLEN     = 0.05
LINE_COLOR       = (0,0,255)               # 红色 (BGR)

# 参考图片尺寸（4096*3072）
REFERENCE_WIDTH = 4096
REFERENCE_HEIGHT = 3072
REFERENCE_CHINESE_FONT_SIZE = 90  # 4096*3072图片对应的中文字体大小
REFERENCE_ENGLISH_FONT_SCALE = 3.0  # 4096*3072图片对应的英文字体缩放

# ============= 字体大小计算 =============
def calculate_font_size_by_image(img_width: int, img_height: int):
    """根据图片物理尺寸估算合适的中英文字体大小"""
    img_area = img_width * img_height
    ref_area = REFERENCE_WIDTH * REFERENCE_HEIGHT
    area_ratio = (img_area / ref_area) ** 0.5  # 使用平方根保持线性视觉比例

    # 将字体大小放大1.2倍
    cn_size = int(REFERENCE_CHINESE_FONT_SIZE * area_ratio * 1.2)
    en_scale = REFERENCE_ENGLISH_FONT_SCALE * area_ratio * 1.2

    cn_size = max(20, min(cn_size, 200))
    en_scale = max(0.5, min(en_scale, 8.0))

    print("🔍 字体大小计算:")
    print(f"  图片尺寸: {img_width}×{img_height}")
    print(f"  中文字体大小: {cn_size} (1.2倍放大)")
    print(f"  英文字体缩放: {en_scale:.2f} (1.2倍放大)")
    return cn_size, en_scale

# ============= 字体加载 =============
FALLBACK_FONT_PATHS = [
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

def get_chinese_font(size: int = 20):
    for fp in FALLBACK_FONT_PATHS:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()

# ============= 绘制中文辅助 =============

def put_chinese_text(img, text, position, font_size=20, color=(0,0,0)):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(pil_img)
    draw.text(position, text, font=get_chinese_font(font_size), fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ============= OCR JSON 读取 =============

def load_ocr_result(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

# ============= 方位辅助 =============

def sector_of(dx: float, dy: float):
    ang = (math.degrees(math.atan2(-dy, dx)) + 360) % 360
    idx = int((ang + 22.5) // 45) % 8
    return SECTORS[idx]

def map_to_basic_region(direction):
    """将8种方向映射到4种基本区域"""
    if direction in ['N', 'NE', 'NW']:
        return 'N'
    elif direction in ['S', 'SE', 'SW']:
        return 'S'
    elif direction in ['E', 'NE', 'SE']:
        return 'E'
    elif direction in ['W', 'NW', 'SW']:
        return 'W'
    else:
        return direction  # 如果已经是基本方向，直接返回

# ============= 槽位生成 =============

def prepare_slots(img_w: int, img_h: int, margin: int, *, is_portrait: bool=False, font_size: int=28):
    """生成可用文字槽位集合，基于四个区域（N、S、E、W）"""
    
    # 根据字体大小动态设置槽位参数
    min_distance_from_image = font_size * 2 # 与原图的最小距离 = 2倍字体大小
    slot_width = font_size * 10 # 槽位宽度 = 16倍字体大小
    slot_height = font_size * 2  # 槽位高度 = 2倍字体大小
    
    # 区分N/S和E/W区域的槽位间距
    # N/S区域：横向间距 = 槽位宽度 + 字体大小，纵向间距 = 槽位高度 + 字体大小
    ns_horizontal_spacing = slot_width + font_size  # N/S区域横向间距
    ns_vertical_spacing = slot_height + font_size   # N/S区域纵向间距
    
    # E/W区域：横向间距 = 槽位宽度 + 字体大小，纵向间距 = 槽位高度 + 字体大小
    ew_horizontal_spacing = slot_width + font_size  # E/W区域横向间距
    ew_vertical_spacing = slot_height + font_size   # E/W区域纵向间距
    print(f"字体大小:{font_size} ######################################")
    
    slots = defaultdict(list)
    
    # 原图边界
    img_left = margin
    img_right = margin + img_w
    img_top = margin
    img_bottom = margin + img_h
    
    # ─── N区域（原图上方） ────────────────────────────────────────────
    y_n = img_top - min_distance_from_image - slot_height
    while y_n >= 10:  # 确保在画布范围内
        # 在N区域水平分布多个槽位
        for x in range(30, img_w + 2*margin - slot_width, ns_horizontal_spacing):
            if x + slot_width <= img_w + 2*margin - 10:
                slots['N'].append((x, y_n, slot_width, slot_height))
        y_n -= (slot_height + ns_vertical_spacing)
    
    # 如果N区域没有槽位，增加拓展距离
    if not slots['N']:
        print("⚠️ N区域没有槽位，增加拓展距离")
        y_n = img_top - min_distance_from_image * 2 - slot_height
        while y_n >= 10:
            for x in range(30, img_w + 2*margin - slot_width, ns_horizontal_spacing):
                if x + slot_width <= img_w + 2*margin - 10:
                    slots['N'].append((x, y_n, slot_width, slot_height))
            y_n -= (slot_height + ns_vertical_spacing)
    
    # ─── S区域（原图下方） ────────────────────────────────────────────
    y_s = img_bottom + min_distance_from_image
    while y_s + slot_height <= img_h + 2*margin - 10:  # 确保在画布范围内
        # 在S区域水平分布多个槽位
        for x in range(30, img_w + 2*margin - slot_width, ns_horizontal_spacing):
            if x + slot_width <= img_w + 2*margin - 10:
                slots['S'].append((x, y_s, slot_width, slot_height))
        y_s += (slot_height + ns_vertical_spacing)
    
    # 如果S区域没有槽位，增加拓展距离
    if not slots['S']:
        print("⚠️ S区域没有槽位，增加拓展距离")
        y_s = img_bottom + min_distance_from_image * 2
        while y_s + slot_height <= img_h + 2*margin - 10:
            for x in range(30, img_w + 2*margin - slot_width, ns_horizontal_spacing):
                if x + slot_width <= img_w + 2*margin - 10:
                    slots['S'].append((x, y_s, slot_width, slot_height))
            y_s += (slot_height + ns_vertical_spacing)
    
    # ─── W区域（原图左侧） ────────────────────────────────────────────
    x_w = img_left - min_distance_from_image - slot_width
    while x_w >= 10:  # 确保在画布范围内
        # 在W区域垂直分布多个槽位，y轴范围限制在N区域下界到S区域上界
        y_start = img_top + slot_height # N区域下界
        y_end = img_bottom - slot_height # S区域上界
        for y in range(max(10, y_start), min(y_end, img_h + 2*margin - slot_height), ew_vertical_spacing):
            if y + slot_height <= img_h + 2*margin - 10:
                slots['W'].append((x_w, y, slot_width, slot_height))
        x_w -= (slot_width + ew_horizontal_spacing)
    
    # 如果W区域没有槽位，增加拓展距离
    if not slots['W']:
        print("⚠️ W区域没有槽位，增加横向拓展距离")
        # 增大横向拓展距离，确保有足够空间
        x_w = img_left - min_distance_from_image * 4 - slot_width
        while x_w >= 10:
            y_start = img_top + slot_height
            y_end = img_bottom - slot_height
            for y in range(max(10, y_start), min(y_end, img_h + 2*margin - slot_height), ew_vertical_spacing):
                if y + slot_height <= img_h + 2*margin - 10:
                    slots['W'].append((x_w, y, slot_width, slot_height))
            x_w -= (slot_width + ew_horizontal_spacing)
    
    # ─── E区域（原图右侧） ────────────────────────────────────────────
    x_e = img_right + min_distance_from_image
    while x_e + slot_width <= img_w + 2*margin - 10:  # 确保在画布范围内
        # 在E区域垂直分布多个槽位，y轴范围限制在N区域下界到S区域上界
        y_start = img_top + slot_height # N区域下界
        y_end = img_bottom - slot_height  # S区域上界
        for y in range(max(10, y_start), min(y_end, img_h + 2*margin - slot_height), ew_vertical_spacing):
            if y + slot_height <= img_h + 2*margin - 10:
                slots['E'].append((x_e, y, slot_width, slot_height))
        x_e += (slot_width + ew_horizontal_spacing)
    
    # 如果E区域没有槽位，增加拓展距离
    if not slots['E']:
        print("⚠️ E区域没有槽位，增加横向拓展距离")
        # 增大横向拓展距离，确保有足够空间
        x_e = img_right + min_distance_from_image * 4
        while x_e + slot_width <= img_w + 2*margin - 10:
            y_start = img_top + slot_height
            y_end = img_bottom - slot_height
            for y in range(max(10, y_start), min(y_end, img_h + 2*margin - slot_height), ew_vertical_spacing):
                if y + slot_height <= img_h + 2*margin - 10:
                    slots['E'].append((x_e, y, slot_width, slot_height))
            x_e += (slot_width + ew_horizontal_spacing)

    # Debug
    print("🔍 槽位生成:")
    for d,lst in slots.items():
        print(f"  {d}: {len(lst)} 个槽位")
        if lst:
            x, y, w, h = lst[0]
            region_name = {'N': '上方', 'S': '下方', 'W': '左侧', 'E': '右侧'}[d]
            print(f"    示例槽位: ({x}, {y}, {w}, {h}) - 位置: {region_name}")

    return slots

# ============= 槽位分配 =============

def optimize_region_selection(arrow_direction, used):
    """
    根据箭头方向和已使用槽位数量优化区域选择
    
    Args:
        arrow_direction: 箭头方向（8个方向）
        used: 已使用槽位计数 {'N': 2, 'S': 1, 'E': 0, 'W': 3}
    
    Returns:
        preferred_regions: 优化后的区域优先级列表
    """
    # 根据箭头方向确定相关区域
    direction_to_regions = {
        'N': ['N'],
        'S': ['S'], 
        'E': ['E'],
        'W': ['W'],
        'NE': ['N', 'E'],  # 东北方向优先选择N和E区域
        'NW': ['N', 'W'],  # 西北方向优先选择N和W区域
        'SE': ['S', 'E'],  # 东南方向优先选择S和E区域
        'SW': ['S', 'W']   # 西南方向优先选择S和W区域
    }
    
    # 获取相关区域
    related_regions = direction_to_regions.get(arrow_direction, ['N', 'S', 'E', 'W'])
    
    # 按已使用槽位数量排序（少的优先）
    sorted_regions = sorted(related_regions, key=lambda region: used.get(region, 0))
    
    # 添加其他区域作为备选
    all_regions = ['N', 'S', 'E', 'W']
    remaining_regions = [r for r in all_regions if r not in sorted_regions]
    remaining_regions.sort(key=lambda region: used.get(region, 0))
    
    preferred_regions = sorted_regions + remaining_regions
    
    print(f"箭头方向: {arrow_direction}, 相关区域: {related_regions}, 已使用: {dict(used)}")
    print(f"优化后优先级: {preferred_regions}")
    
    return preferred_regions

def is_overlap(x1, y1, w1, h1, x2, y2, w2, h2):
    """检查两个矩形是否重叠"""
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

def assign_translation(block_ctr, centroid, slots, used, *, is_portrait=False, img_w=0, img_h=0, margin=0, drawn_blocks=None, font_size=28, translation_text=""):
    """
    分配翻译槽位，避免重叠并确保在原图外
    
    Args:
        block_ctr: OCR块中心位置
        centroid: 图片中心位置
        slots: 可用槽位
        used: 已使用槽位计数
        is_portrait: 是否为竖图
        img_w: 原图宽度
        img_h: 原图高度
        margin: 边距
        drawn_blocks: 已绘制的翻译块范围列表 [(x1,y1,x2,y2), ...]
        font_size: 字体大小，用于预估框尺寸
        translation_text: 翻译文本，用于准确预估框尺寸
    """
    # 以原图中心为基准点，计算待翻译框相对于原图中心的方向（仅用于箭头绘制）
    dx, dy = block_ctr[0] - centroid[0], block_ctr[1] - centroid[1]
    arrow_direction = sector_of(dx, dy)
    
    print(f"待翻译框中心: {block_ctr}, 原图中心: {centroid}, 方向向量: ({dx}, {dy}), 箭头方向: {arrow_direction}, 翻译文本: '{translation_text}'")

    # 原图在扩展后图片中的位置
    img_left = margin
    img_right = margin + img_w
    img_top = margin
    img_bottom = margin + img_h

    def is_inside_image(x, y, w, h):
        """检查槽位是否在原图内"""
        # 检查是否与原图有重叠
        return not (x >= img_right or x + w <= img_left or y >= img_bottom or y + h <= img_top)

    def find_non_overlapping_slot(preferred_directions):
        """在指定方向中寻找不重叠的槽位"""
        print(f"🔍 寻找槽位，首选方向: {preferred_directions}")
        
        # 对每个首选方向，尝试该方向的所有可用槽位
        for direction in preferred_directions:
            print(f"  尝试方向 {direction}: 已使用 {used[direction]}/{len(slots[direction])} 个槽位")
            
            # 尝试该方向的所有可用槽位
            for slot_idx in range(used[direction], len(slots[direction])):
                slot = slots[direction][slot_idx]
                x, y, w, h = slot
                print(f"    尝试槽位 {slot_idx}: 位置 ({x}, {y}), 尺寸 ({w}, {h})")
                
                # 检查是否在原图内
                if is_inside_image(x, y, estimated_box_width, estimated_box_height):
                    print(f"      ❌ 槽位在原图内，跳过")
                    continue
                
                # 检查是否与已绘制的块重叠
                overlap = False
                if drawn_blocks:
                    for drawn_x, drawn_y, drawn_w, drawn_h in drawn_blocks:
                        if is_overlap(x, y, estimated_box_width, estimated_box_height, drawn_x, drawn_y, drawn_w, drawn_h):
                            overlap = True
                            print(f"      ❌ 与已绘制块重叠，跳过")
                            break
                
                if not overlap:
                    used[direction] = slot_idx + 1  # 更新为下一个槽位
                    print(f"      ✅ 选择槽位: {direction} 位置 ({x}, {y})")
                    
                    # 计算在该区域内的行/列位置
                    # 对于N/S区域：计算列位置（水平方向）
                    # 对于E/W区域：计算行位置（垂直方向）
                    if direction in ['N', 'S']:
                        # N/S区域：计算列位置
                        # 计算每行的槽位个数
                        slots_in_direction = slots[direction]
                        if slots_in_direction:
                            # 按y坐标分组，计算每行的槽位个数
                            rows = {}
                            for slot_x, slot_y, slot_w, slot_h in slots_in_direction:
                                if slot_y not in rows:
                                    rows[slot_y] = []
                                rows[slot_y].append((slot_x, slot_y, slot_w, slot_h))
                            
                            # 找到当前槽位所在的行
                            current_row = None
                            for row_y, row_slots in rows.items():
                                for row_slot in row_slots:
                                    if abs(row_slot[0] - x) < 5 and abs(row_slot[1] - y) < 5:  # 找到当前槽位
                                        current_row = row_y
                                        break
                                if current_row is not None:
                                    break
                            
                            if current_row is not None:
                                # 计算在当前行中的位置
                                row_slots = rows[current_row]
                                row_slots.sort(key=lambda s: s[0])  # 按x坐标排序
                                current_slot_in_row = None
                                for i, row_slot in enumerate(row_slots):
                                    if abs(row_slot[0] - x) < 5 and abs(row_slot[1] - y) < 5:
                                        current_slot_in_row = i
                                        break
                                
                                if current_slot_in_row is not None:
                                    col_position = current_slot_in_row / max(1, len(row_slots) - 1)
                                    return (x, y, estimated_box_width, estimated_box_height), direction, col_position
                        
                        # 如果无法确定，使用默认计算
                        total_slots = len(slots[direction])
                        col_position = slot_idx / max(1, total_slots - 1)
                        return (x, y, estimated_box_width, estimated_box_height), direction, col_position
                    else:
                        # E/W区域：计算行位置
                        # 计算每列的槽位个数
                        slots_in_direction = slots[direction]
                        if slots_in_direction:
                            # 按x坐标分组，计算每列的槽位个数
                            cols = {}
                            for slot_x, slot_y, slot_w, slot_h in slots_in_direction:
                                if slot_x not in cols:
                                    cols[slot_x] = []
                                cols[slot_x].append((slot_x, slot_y, slot_w, slot_h))
                            
                            # 找到当前槽位所在的列
                            current_col = None
                            for col_x, col_slots in cols.items():
                                for col_slot in col_slots:
                                    if abs(col_slot[0] - x) < 5 and abs(col_slot[1] - y) < 5:  # 找到当前槽位
                                        current_col = col_x
                                        break
                                if current_col is not None:
                                    break
                            
                            if current_col is not None:
                                # 计算在当前列中的位置
                                col_slots = cols[current_col]
                                col_slots.sort(key=lambda s: s[1])  # 按y坐标排序
                                current_slot_in_col = None
                                for i, col_slot in enumerate(col_slots):
                                    if abs(col_slot[0] - x) < 5 and abs(col_slot[1] - y) < 5:
                                        current_slot_in_col = i
                                        break
                                
                                if current_slot_in_col is not None:
                                    row_position = current_slot_in_col / max(1, len(col_slots) - 1)
                                    return (x, y, estimated_box_width, estimated_box_height), direction, row_position
                        
                        # 如果无法确定，使用默认计算
                        total_slots = len(slots[direction])
                        row_position = slot_idx / max(1, total_slots - 1)
                        return (x, y, estimated_box_width, estimated_box_height), direction, row_position
            
            print(f"  方向 {direction} 的所有槽位都已尝试完毕")
        
        print("❌ 所有首选方向都没有可用槽位")
        return None, None

    # 基于实际翻译文本预估翻译框的尺寸
    estimated_margin = max(30, font_size // 2)
    
    # 计算文本宽度（基于字符数量）
    if translation_text:
        # 中文字符宽度约为字体大小，英文字符宽度约为字体大小的一半
        char_width = font_size if any('\u4e00' <= char <= '\u9fff' for char in translation_text) else font_size // 2
        estimated_text_width = len(translation_text) * char_width
        # 限制最大宽度
        estimated_text_width = min(estimated_text_width, 600)
    else:
        estimated_text_width = 400  # 默认宽度
    
    # 计算文本高度（基于换行）
    if translation_text:
        # 估算换行数量
        line_count = max(1, len(translation_text) // 20)  # 每行约20个字符
        estimated_text_height = line_count * font_size * 1.5  # 行高为字体大小的1.5倍
    else:
        estimated_text_height = font_size * 2  # 默认高度
    
    estimated_box_width = estimated_text_width + 2 * estimated_margin
    estimated_box_height = estimated_text_height + 2 * estimated_margin
    
    # 根据待翻译框位置确定槽位区域（N、S、E、W四个区域）
    block_x, block_y = block_ctr[0], block_ctr[1]
    
    # 确定槽位区域：基于待翻译框相对于原图的位置
    if block_y < img_top:
        # 待翻译框在原图上方 → N区域
        slot_region = 'N'
    elif block_y > img_bottom:
        # 待翻译框在原图下方 → S区域
        slot_region = 'S'
    elif block_x < img_left:
        # 待翻译框在原图左侧 → W区域
        slot_region = 'W'
    else:
        # 待翻译框在原图右侧 → E区域
        slot_region = 'E'
    
    print(f"待翻译框位置: ({block_x}, {block_y}), 原图边界: 左{img_left} 右{img_right} 上{img_top} 下{img_bottom}, 分配区域: {slot_region}")
    
    # 优化优先级选择：根据箭头方向选择最优区域
    preferred_regions = optimize_region_selection(arrow_direction, used)
    
    # 尝试在优化后的区域中找到不重叠的槽位
    slot_result = find_non_overlapping_slot(preferred_regions)
    if slot_result:
        slot, direction, position = slot_result
        # 返回实际分配的区域，而不是映射后的方向
        return slot, direction, position  # 返回实际分配的区域用于绘制

    # 如果所有预生成槽位都不合适，根据槽位区域创建新的槽位
    block_x, block_y = block_ctr[0], block_ctr[1]
    
    # 根据槽位区域创建新槽位
    if slot_region == 'E':
        # 东侧区域 → 创建东侧槽位
        x = block_x + 250 + used['E'] * (estimated_box_width + 100)
        y = block_y - estimated_box_height // 2 + used['E'] * (estimated_box_height + 30)
        direction = 'E'
    elif slot_region == 'W':
        # 西侧区域 → 创建西侧槽位
        x = block_x - 250 - used['W'] * (estimated_box_width + 100)
        y = block_y - estimated_box_height // 2 + used['W'] * (estimated_box_height + 30)
        direction = 'W'
    elif slot_region == 'N':
        # 北侧区域 → 创建北侧槽位
        x = block_x - estimated_box_width // 2 + used['N'] * (estimated_box_width + 100)
        y = block_y - 250 - used['N'] * (estimated_box_height + 30)
        direction = 'N'
    elif slot_region == 'S':
        # 南侧区域 → 创建南侧槽位
        x = block_x - estimated_box_width // 2 + used['S'] * (estimated_box_width + 100)
        y = block_y + 250 + used['S'] * (estimated_box_height + 30)
        direction = 'S'
    else:
        # 默认情况
        x = block_x + 200
        y = block_y - estimated_box_height // 2
        direction = 'E'

    # 边界检查和调整
    canvas_width = img_w + 2 * margin
    canvas_height = img_h + 2 * margin
    
    # 确保在原图外
    if is_inside_image(x, y, estimated_box_width, estimated_box_height):
        if x < img_right:  # 如果在原图左侧，移到左侧
            x = img_left - estimated_box_width - 50
            direction = 'W'  # 调整到左侧，方向改为西
        else:  # 如果在原图右侧，移到右侧
            x = img_right + 50
            direction = 'E'  # 调整到右侧，方向改为东
        print(f"⚠️ 槽位调整到原图外: ({x}, {y}), 方向调整为: {direction}")
    
    # 边界检查
    x = max(10, min(x, canvas_width - estimated_box_width - 10))
    y = max(10, min(y, canvas_height - estimated_box_height - 10))
    
    # 检查是否与已绘制块重叠
    if drawn_blocks:
        for drawn_x, drawn_y, drawn_w, drawn_h in drawn_blocks:
            if is_overlap(x, y, estimated_box_width, estimated_box_height, drawn_x, drawn_y, drawn_w, drawn_h):
                # 如果重叠，向下移动
                y = drawn_y + drawn_h + 20
                y = min(y, canvas_height - estimated_box_height - 10)  # 确保不超出边界
                print(f"⚠️ 检测到重叠，调整Y位置到: {y}")
                break
    
    # 根据最终位置重新计算方向
    final_center_x = x + estimated_box_width // 2
    final_center_y = y + estimated_box_height // 2
    final_dx = final_center_x - centroid[0]
    final_dy = final_center_y - centroid[1]
    final_direction = sector_of(final_dx, final_dy)
    
    # 将最终方向映射到基本区域
    final_basic_region = map_to_basic_region(final_direction)
    
    # 如果方向发生了变化，更新方向
    if final_basic_region != direction:
        print(f"⚠️ 方向从 {direction} 调整为: {final_basic_region} (原8方向: {final_direction})")
        direction = final_basic_region
    
    used[direction] += 1
    
    # 计算在该区域内的行/列位置
    # 对于N/S区域：计算列位置（水平方向）
    # 对于E/W区域：计算行位置（垂直方向）
    if direction in ['N', 'S']:
        # N/S区域：计算列位置
        # 计算每行的槽位个数
        slots_in_direction = slots.get(direction, [])
        if slots_in_direction:
            # 按y坐标分组，计算每行的槽位个数
            rows = {}
            for slot_x, slot_y, slot_w, slot_h in slots_in_direction:
                if slot_y not in rows:
                    rows[slot_y] = []
                rows[slot_y].append((slot_x, slot_y, slot_w, slot_h))
            
            # 找到当前槽位所在的行（基于位置）
            current_row = None
            for row_y, row_slots in rows.items():
                for row_slot in row_slots:
                    if abs(row_slot[0] - x) < 5 and abs(row_slot[1] - y) < 5:  # 找到当前槽位
                        current_row = row_y
                        break
                if current_row is not None:
                    break
            
            if current_row is not None:
                # 计算在当前行中的位置
                row_slots = rows[current_row]
                row_slots.sort(key=lambda s: s[0])  # 按x坐标排序
                current_slot_in_row = None
                for i, row_slot in enumerate(row_slots):
                    if abs(row_slot[0] - x) < 5 and abs(row_slot[1] - y) < 5:
                        current_slot_in_row = i
                        break
                
                if current_slot_in_row is not None:
                    col_position = current_slot_in_row / max(1, len(row_slots) - 1)
                    return (x, y, estimated_box_width, estimated_box_height), direction, col_position
        
        # 如果无法确定，使用默认计算
        total_slots = len(slots.get(direction, []))
        col_position = (used[direction] - 1) / max(1, total_slots - 1)
        return (x, y, estimated_box_width, estimated_box_height), direction, col_position
    else:
        # E/W区域：计算行位置
        # 计算每列的槽位个数
        slots_in_direction = slots.get(direction, [])
        if slots_in_direction:
            # 按x坐标分组，计算每列的槽位个数
            cols = {}
            for slot_x, slot_y, slot_w, slot_h in slots_in_direction:
                if slot_x not in cols:
                    cols[slot_x] = []
                cols[slot_x].append((slot_x, slot_y, slot_w, slot_h))
            
            # 找到当前槽位所在的列（基于位置）
            current_col = None
            for col_x, col_slots in cols.items():
                for col_slot in col_slots:
                    if abs(col_slot[0] - x) < 5 and abs(col_slot[1] - y) < 5:  # 找到当前槽位
                        current_col = col_x
                        break
                if current_col is not None:
                    break
            
            if current_col is not None:
                # 计算在当前列中的位置
                col_slots = cols[current_col]
                col_slots.sort(key=lambda s: s[1])  # 按y坐标排序
                current_slot_in_col = None
                for i, col_slot in enumerate(col_slots):
                    if abs(col_slot[0] - x) < 5 and abs(col_slot[1] - y) < 5:
                        current_slot_in_col = i
                        break
                
                if current_slot_in_col is not None:
                    row_position = current_slot_in_col / max(1, len(col_slots) - 1)
                    return (x, y, estimated_box_width, estimated_box_height), direction, row_position
        
        # 如果无法确定，使用默认计算
        total_slots = len(slots.get(direction, []))
        row_position = (used[direction] - 1) / max(1, total_slots - 1)
        return (x, y, estimated_box_width, estimated_box_height), direction, row_position

def arrow_endpoint(dir_, x, y, w, h, relative_position=0):
    """
    箭头指向框的反方向，只处理N、S、E、W四种基本方向
    
    Args:
        dir_: 方向 ('N', 'S', 'E', 'W')
        x, y, w, h: 框的位置和尺寸
        relative_position: 该框在其区域内的相对位置 (0到1之间的比例)
    """
    if   dir_ == 'N':  return int(x + w - relative_position * w), int(y + h)  # 北侧框 → 指向框的南边（下边）
    elif dir_ == 'S':  return int(x + w - relative_position * w), int(y)       # 南侧框 → 指向框的北边（上边）
    elif dir_ == 'E':  return int(x), int(y + h - relative_position * h)       # 东侧框 → 指向框的西边（左边）
    elif dir_ == 'W':  return int(x + w), int(y + h - relative_position * h)   # 西侧框 → 指向框的东边（右边）
    else:              
        return int(x + w // 2), int(y + h // 2)  # 默认情况

# ============= 文本绘制（支持中文） =============
def draw_multiline(img, text, topleft, font, scale, thk, color, max_w):
    x0, y0 = topleft
    
    # 检查是否包含中文字符
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
    
    if has_chinese:
        # 使用PIL绘制中文
        return draw_multiline_chinese(img, text, topleft, max_w, color)
    else:
        # 使用OpenCV绘制英文
        return draw_multiline_opencv(img, text, topleft, font, scale, thk, color, max_w)

def draw_multiline_chinese(img, text, topleft, max_w, color):
    """使用PIL绘制多行中文文字"""
    try:
        # 转换为PIL图像
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # 获取字体
        font = get_chinese_font(20)  # 使用20号字体
        
        x0, y0 = topleft
        
        # 简单的换行处理
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            # 估算文字宽度（简单估算）
            estimated_width = len(test_line) * 15  # 每个字符约15像素
            
            if estimated_width <= max_w:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        # 绘制每一行
        line_height = 25
        for i, line in enumerate(lines):
            draw.text((x0, y0 + i * line_height), line, fill=color, font=font)
        
        # 转换回OpenCV格式
        result_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # 计算实际宽度和高度
        total_height = len(lines) * line_height
        max_line_width = max(len(line) * 15 for line in lines) if lines else 0
        
        return max_line_width, total_height
        
    except Exception as e:
        print(f"绘制中文文字失败: {e}")
        # 如果失败，使用OpenCV默认方法
        return draw_multiline_opencv(img, text, topleft, cv2.FONT_HERSHEY_SIMPLEX, 1, 2, color, max_w)

def draw_multiline_opencv(img, text, topleft, font, scale, thk, color, max_w):
    """使用OpenCV绘制多行英文文字"""
    x0, y0 = topleft
    words = text.split()
    lines = []
    while words:
        cur = []
        while words:
            trial = ' '.join(cur + [words[0]])
            tw, _ = cv2.getTextSize(trial, font, scale, thk)[0]
            if tw <= max_w:
                cur.append(words.pop(0))
            else:
                break
        if not cur:                    # 单词本身太长，强制拆
            cur = [words.pop(0)]
        lines.append(' '.join(cur))

    line_h = cv2.getTextSize('A', font, scale, thk)[0][1] + font_size // 2
    for i, l in enumerate(lines):
        cv2.putText(img, l, (x0, y0 + (i + 1) * line_h), font, scale, color, thk)

    total_h = line_h * len(lines)
    return max_w, total_h

# ============= 中文翻译绘制函数 =============
def draw_chinese_translation(img, text, position, font_size=28, color=(0, 0, 0), max_width=760):
    """
    专门绘制中文翻译的函数
    
    Args:
        img: OpenCV图像
        text: 要绘制的中文文本
        position: 绘制位置 (x, y)
        font_size: 字体大小，默认24
        color: 文字颜色，默认黑色
        max_width: 最大宽度，默认760
    
    Returns:
        tuple: (实际宽度, 实际高度)
    """
    try:
        # 转换为PIL图像
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # 获取中文字体
        font = get_chinese_font(font_size)
        
        x0, y0 = position
        
        # 中文文本换行处理
        lines = []
        current_line = ""
        
        # 按字符分割，支持中英文混合
        chars = list(text)
        for char in chars:
            test_line = current_line + char
            
            # 使用实际字体大小估算文字宽度
            estimated_width = 0
            for c in test_line:
                if '\u4e00' <= c <= '\u9fff':  # 中文字符
                    estimated_width += font_size
                elif c.isupper():  # 大写英文字符
                    estimated_width += font_size * 0.6
                elif c.islower():  # 小写英文字符
                    estimated_width += font_size * 0.5
                elif c.isdigit():  # 数字
                    estimated_width += font_size * 0.6
                else:  # 其他字符
                    estimated_width += font_size * 0.4
            
            if estimated_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = char
        
        if current_line:
            lines.append(current_line)
        
        # 如果没有内容，返回默认尺寸
        if not lines:
            return 0, 0
        
        # 绘制每一行
        line_height = font_size + int(font_size * 0.3)  # 根据字体大小动态调整行间距
        max_line_width = 0
        
        for i, line in enumerate(lines):
            # 绘制文字
            draw.text((x0, y0 + i * line_height), line, fill=color, font=font)
            
            # 计算当前行的实际宽度
            line_width = 0
            for c in line:
                if '\u4e00' <= c <= '\u9fff':  # 中文字符
                    line_width += font_size
                elif c.isupper():  # 大写英文字符
                    line_width += font_size * 0.6
                elif c.islower():  # 小写英文字符
                    line_width += font_size * 0.5
                elif c.isdigit():  # 数字
                    line_width += font_size * 0.6
                else:  # 其他字符
                    line_width += font_size * 0.4
            max_line_width = max(max_line_width, line_width)
        
        # 转换回OpenCV格式
        result_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # 更新原图像
        img[:] = result_img[:]
        
        # 返回实际尺寸
        total_height = len(lines) * line_height
        return max_line_width, total_height
        
    except Exception as e:
        print(f"绘制中文翻译失败: {e}")
        # 如果失败，使用OpenCV默认方法作为后备
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return len(text) * 20, 30  # 估算尺寸

def draw_multiline_chinese_enhanced(img, text, topleft, max_w, color, font_size=28):
    """
    增强版中文多行绘制函数
    
    Args:
        img: OpenCV图像
        text: 要绘制的中文文本
        topleft: 绘制位置 (x, y)
        max_w: 最大宽度
        color: 文字颜色
        font_size: 字体大小
    
    Returns:
        tuple: (实际宽度, 实际高度)
    """
    try:
        # 转换为PIL图像
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # 获取中文字体
        font = get_chinese_font(font_size)
        
        x0, y0 = topleft
        
        # 智能换行处理
        lines = []
        current_line = ""
        
        # 按字符分割，支持中英文混合
        chars = list(text)
        for char in chars:
            test_line = current_line + char
            
            # 更精确的宽度估算
            estimated_width = 0
            for c in test_line:
                if '\u4e00' <= c <= '\u9fff':  # 中文字符
                    estimated_width += font_size
                elif c.isupper():  # 大写英文字符
                    estimated_width += font_size * 0.6
                elif c.islower():  # 小写英文字符
                    estimated_width += font_size * 0.5
                elif c.isdigit():  # 数字
                    estimated_width += font_size * 0.6
                else:  # 其他字符
                    estimated_width += font_size * 0.4
            
            if estimated_width <= max_w:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = char
        
        if current_line:
            lines.append(current_line)
        
        # 如果没有内容，返回默认尺寸
        if not lines:
            return 0, 0
        
        # 绘制每一行
        line_height = font_size + int(font_size * 0.3)  # 根据字体大小动态调整行间距
        max_line_width = 0
        
        for i, line in enumerate(lines):
            # 绘制文字
            draw.text((x0, y0 + i * line_height), line, fill=color, font=font)
            
            # 计算当前行的实际宽度
            line_width = 0
            for c in line:
                if '\u4e00' <= c <= '\u9fff':  # 中文字符
                    line_width += font_size
                elif c.isupper():  # 大写英文字符
                    line_width += font_size * 0.6
                elif c.islower():  # 小写英文字符
                    line_width += font_size * 0.5
                elif c.isdigit():  # 数字
                    line_width += font_size * 0.6
                else:  # 其他字符
                    line_width += font_size * 0.4
            
            max_line_width = max(max_line_width, line_width)
        
        # 转换回OpenCV格式
        result_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # 更新原图像
        img[:] = result_img[:]
        
        # 返回实际尺寸
        total_height = len(lines) * line_height
        return max_line_width, total_height
        
    except Exception as e:
        print(f"增强版中文绘制失败: {e}")
        # 如果失败，使用OpenCV默认方法作为后备
        cv2.putText(img, text, topleft, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return len(text) * 20, 30  # 估算尺寸

# ============= 翻译解析 =============
def parse_ai(ai_txt):
    trans, merged, regions = {}, set(), {}
    for ln in ai_txt.strip().split('\n'):
        if '[' not in ln or '->' not in ln:
            continue
        num_part = ln[ln.find('[') + 1: ln.find(']')]
        txt = ln.split('->', 1)[1].strip()
        if '-' in num_part:
            a, b = map(int, num_part.split('-'))
            trans[a] = txt
            regions[a] = {'translation': txt, 'start_block': a, 'end_block': b}
            merged.update(range(a + 1, b + 1))
        else:
            trans[int(num_part)] = txt
    return trans, merged, regions

# ============= 主函数（保持原名字便于 import） =============
def visualize_ocr_results_with_translation(img_path, json_path, ai_resp,
                                           out_path="output/vis_final.jpg",
                                           visualization_language='chinese'):
    # ---------- 读取 ----------
    if (img := cv2.imread(img_path)) is None:
        raise FileNotFoundError(img_path)
    ocr = load_ocr_result(json_path)
    rec = zip(ocr['rec_texts'], ocr['rec_scores'], ocr['rec_boxes'])
    valid = [{'index': i + 1, 'text': t, 'score': s, 'box': b}
             for i, (t, s, b) in enumerate(rec) if t.strip() and s > 0.5]
    for i in range(len(valid)):
        valid[i]['index'] = i + 1
    

    trans, merged, regions = parse_ai(ai_resp)
    
    # 调试信息
    print("🔍 解析结果调试信息:")
    print(f"  翻译字典: {trans}")
    print(f"  合并集合: {merged}")
    print(f"  区域信息: {regions}")
    print(f"  有效OCR块数量: {len(valid)}")
    print("  OCR块内容:")
    for i, v in enumerate(valid[:10]):  # 只显示前10个
        print(f"    块 {v['index']}: '{v['text']}'")
    
    # 显示翻译匹配情况
    print("  翻译匹配情况:")
    for i in range(1, min(11, len(valid) + 1)):
        if i in trans:
            print(f"    块 {i} -> 直接翻译: '{trans[i]}'")
        elif i in regions:
            print(f"    块 {i} -> 合并区域翻译: '{regions[i]['translation']}'")
        elif i in merged:
            print(f"    块 {i} -> 被合并到其他块")
        else:
            print(f"    块 {i} -> 无翻译")

    # ---------- 画布 ----------
    h, w = img.shape[:2]
    
    # 根据图片尺寸计算字体大小
    chinese_font_size, english_font_scale = calculate_font_size_by_image(w, h)
    

    
    # 动态计算边距，确保翻译文字在原图外
    base_margin = max(200, w // 4)
    
    # 针对竖图（height > width）的特殊处理
    is_portrait = h > w
    if is_portrait:
        print(f"📱 检测到竖图: {w}x{h}，将增加宽度方向扩展区域")
        # 竖图需要更大的宽度边距来避免文字重叠
        base_margin = max(300, w // 2)  # 竖图基础边距更大
    
    # 根据翻译文字的数量和长度动态调整边距
    translation_count = len([v for v in valid if v['index'] in trans or v['index'] in regions])
    if translation_count > 0:
        # 计算平均翻译文字长度
        total_translation_length = 0
        translation_chars = 0
        for v in valid:
            idx = v['index']
            if idx in trans:
                total_translation_length += len(trans[idx])
                translation_chars += 1
            elif idx in regions:
                total_translation_length += len(regions[idx]['translation'])
                translation_chars += 1
        
        if translation_chars > 0:
            avg_translation_length = total_translation_length / translation_chars
            # 根据平均长度调整边距
            if avg_translation_length > 20:
                margin = max(base_margin, w * 1.5) if is_portrait else max(base_margin, w)  # 竖图长文字，增加更多边距
            elif avg_translation_length > 10:
                margin = max(base_margin, w * 1.2) if is_portrait else max(base_margin, w * 0.8)  # 竖图中等文字
            else:
                margin = base_margin
        else:
            margin = base_margin
    else:
        margin = base_margin
    
    original_margin = margin
    # 确保边距足够大，能够容纳翻译文字
    margin = max(margin, 800) if is_portrait else max(margin, 600)  # 增加最小边距
    
    # 根据可视化语言类型确定字体大小
    if visualization_language == 'chinese':
        font_size = chinese_font_size
    else:
        font_size = int(english_font_scale * 20)
    
    # 计算槽位参数
    slot_width = font_size * 10
    slot_height = font_size * 2
    ns_horizontal_spacing = slot_width + font_size
    ns_vertical_spacing = slot_height + font_size
    ew_horizontal_spacing = slot_width + font_size
    ew_vertical_spacing = slot_height + font_size
    

    
    # 检查N/S方向是否有足够空间放置槽位
    def check_ns_slots_available(margin):
        # N方向需要的空间
        y_n = margin - min_distance_from_image - slot_height
        if y_n < 10:  # 如果N方向没有空间
            return False
        
        # S方向需要的空间
        y_s = margin + h + min_distance_from_image
        if y_s + slot_height > h + 2*margin - 10:  # 如果S方向没有空间
            return False
        
        # 检查水平方向是否有足够空间放置至少一个槽位
        available_width = w + 2*margin - 20  # 可用宽度
        if available_width < slot_width:  # 如果连一个槽位都放不下
            return False
        
        return True
    
    # 检查E/W方向是否有足够空间放置槽位
    def check_ew_slots_available(margin):
        # W方向需要的空间
        x_w = margin - min_distance_from_image - slot_width
        if x_w < 10:  # 如果W方向没有空间
            return False
        
        # E方向需要的空间
        x_e = margin + w + min_distance_from_image
        if x_e + slot_width > w + 2*margin - 10:  # 如果E方向没有空间
            return False
        
        # 检查垂直方向是否有足够空间放置至少一个槽位
        available_height = h + 2*margin - 20  # 可用高度
        if available_height < slot_height:  # 如果连一个槽位都放不下
            return False
        
        return True
    
    # 自动增大画布直到有足够空间
    min_distance_from_image = font_size * 2
    while not (check_ns_slots_available(int(margin)) and check_ew_slots_available(int(margin))):
        print(f"⚠️ 当前边距 {margin} 不足以放置槽位，增大画布")
        margin += 100  # 每次增加100像素
        if margin > 2000:  # 防止无限循环
            print("⚠️ 边距过大，停止增大")
            break
    
    print(f"🔍 画布设置:")
    print(f"  原图尺寸: {w}x{h}")
    print(f"  基础边距: {base_margin}")
    print(f"  最终边距: {margin}")
    print(f"  翻译数量: {translation_count}")
    print(f"  槽位参数: 宽度={slot_width}, 高度={slot_height}")
    print(f"  N/S间距: 横向={ns_horizontal_spacing}, 纵向={ns_vertical_spacing}")
    print(f"  E/W间距: 横向={ew_horizontal_spacing}, 纵向={ew_vertical_spacing}")
    
    # 确保margin是整数
    margin = int(margin)
    canv = np.ones((h + 2 * margin, w + 2 * margin, 3), np.uint8) * 240
    canv[margin: margin + h, margin: margin + w] = img

    centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in (v['box'] for v in valid)]
    centroid = (sum(x for x, _ in centers) / len(centers),
                sum(y for _, y in centers) / len(centers))

    slots = prepare_slots(w, h, int(margin), is_portrait=is_portrait, font_size=font_size)
    used_slots = defaultdict(int)
    font = cv2.FONT_HERSHEY_SIMPLEX
    MAX_W = TEXT_W - 20
    


    # 记录已绘制的翻译块范围，用于避免重叠
    drawn_blocks = []
    
    for v, ctr in zip(valid, centers):
        idx, (x1, y1, x2, y2) = v['index'], [int(c + int(margin)) for c in v['box']]
        cv2.rectangle(canv, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 找翻译 - 修正索引偏移问题
        txt = None
        
        # 首先检查是否有直接的翻译
        if idx in trans:
            txt = trans[idx]
        # 然后检查是否在合并区域中
        elif idx in regions:
            txt = regions[idx].get('translation')
        # 最后检查是否被合并到其他区域
        else:
            for region_start, region_info in regions.items():
                if region_info['start_block'] <= idx <= region_info['end_block']:
                    txt = region_info['translation']
                    break
        
        if not txt:
            continue

        # 检查是否是被合并的块（不是合并区域的起始块）
        is_merged_block = False
        for region_start, region_info in regions.items():
            if idx != region_start and region_info['start_block'] <= idx <= region_info['end_block']:
                is_merged_block = True
                break
        
        # 如果是被合并的块，跳过绘制翻译和连线
        if is_merged_block:
            continue

        # 合并区域重新取中心
        if idx in regions:
            a, b = regions[idx]['start_block'], regions[idx]['end_block']
            mx1, my1, mx2, my2 = x1, y1, x2, y2
            for j in range(a, b + 1):
                if j - 1 < len(valid):
                    bx1, by1, bx2, by2 = [c + int(margin) for c in valid[j - 1]['box']]
                    mx1, my1 = min(mx1, bx1), min(my1, by1)
                    mx2, my2 = max(mx2, bx2), max(my2, by2)
            cv2.rectangle(canv, (int(mx1), int(my1)), (int(mx2), int(my2)), (255, 0, 255), 3)
            ctr = ((mx1 + mx2) / 2 - margin, (my1 + my2) / 2 - margin)

        # 计算翻译框的位置和方向
        # 对于合并区域，使用合并后的边界框中心
        if idx in regions:
            # 合并区域：使用合并后的边界框中心
            a, b = regions[idx]['start_block'], regions[idx]['end_block']
            # 计算合并框的边界
            min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
            for j in range(a, b + 1):
                if j - 1 < len(valid):
                    bx1, by1, bx2, by2 = valid[j - 1]['box']
                    min_x = min(min_x, bx1)
                    min_y = min(min_y, by1)
                    max_x = max(max_x, bx2)
                    max_y = max(max_y, by2)
            # 合并框的中心
            merged_center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
            (sx, sy, sw, sh), dir_, position = assign_translation(merged_center, centroid, slots, used_slots, is_portrait=is_portrait, img_w=w, img_h=h, margin=int(margin), drawn_blocks=drawn_blocks, font_size=chinese_font_size if visualization_language == 'chinese' else int(english_font_scale * 20), translation_text=txt)
        else:
            # 单个OCR块：使用原始中心
            (sx, sy, sw, sh), dir_, position = assign_translation(ctr, centroid, slots, used_slots, is_portrait=is_portrait, img_w=w, img_h=h, margin=int(margin), drawn_blocks=drawn_blocks, font_size=chinese_font_size if visualization_language == 'chinese' else int(english_font_scale * 20), translation_text=txt)

        # 文字（支持中文）
        # 检测文本是否包含中文字符
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in txt)
        
        # 根据可视化语言类型和图片尺寸决定字体大小
        if visualization_language == 'chinese':
            # 使用动态计算的中文字体大小
            tw, th = draw_chinese_translation(canv, txt, (sx, sy), font_size=chinese_font_size, color=(0, 0, 0), max_width=MAX_W)
        else:
            # 使用动态计算的英文字体缩放
            tw, th = draw_multiline(canv, txt, (sx, sy), font, english_font_scale, 3, (0, 0, 0), MAX_W)
        
        # 计算边距（与字体大小成比例）
        if visualization_language == 'chinese':
            margin_size = max(30, chinese_font_size // 2)  # 中文字体大，需要更大边距
        else:
            margin_size = max(15, int(english_font_scale * 5))  # 英文字体小，边距也小
        
        # 检查是否与已绘制的块重叠（基于实际文字尺寸）
        actual_box_x = sx - margin_size
        actual_box_y = sy - margin_size
        actual_box_w = tw + 2 * margin_size
        actual_box_h = th + 2 * margin_size
        
        # 背景矩形 - 适应实际文字内容
        cv2.rectangle(canv, (int(sx - margin_size), int(sy - margin_size)), (int(sx + tw + margin_size), int(sy + th + margin_size)), (255, 255, 255), -1)
        
        # 重新绘制文字（因为背景覆盖了文字）
        if visualization_language == 'chinese':
            draw_chinese_translation(canv, txt, (sx, sy), font_size=chinese_font_size, color=(0, 0, 0), max_width=MAX_W)
        else:
            draw_multiline(canv, txt, (sx, sy), font, english_font_scale, 3, (0, 0, 0), MAX_W)
        
        # 文字框边缘 - 适应实际文字内容，确保包住字体
        cv2.rectangle(canv, (int(sx - margin_size), int(sy - margin_size)), (int(sx + tw + margin_size), int(sy + th + margin_size)), (0, 0, 255), 2)

        # 记录已绘制的翻译块范围，用于后续避免重叠
        drawn_blocks.append((sx - margin_size, sy - margin_size, tw + 2 * margin_size, th + 2 * margin_size))

        # 箭头指向框的边缘，而不是文字区域
        # 使用从assign_translation返回的实际位置信息
        print(f"🔍 箭头调试: 方向={dir_}, 位置={position}, 框位置=({sx}, {sy}), 框尺寸=({tw}, {th})")
        ex, ey = arrow_endpoint(dir_, sx - margin_size, sy - margin_size, tw + 2 * margin_size, th + 2 * margin_size, position)
        print(f"  箭头端点: ({ex}, {ey})")
        bx, by = ((x1 + x2) // 2, (y1 + y2) // 2) if idx in trans else \
                 ((mx1 + mx2) // 2, (my1 + my2) // 2)
        cv2.arrowedLine(canv, (int(bx), int(by)), (int(ex), int(ey)), LINE_COLOR,
                        ARROW_THICKNESS, tipLength=ARROW_TIPLEN)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, canv)
    print("✅ 结果已保存到:", out_path)


# ============= 快速测试入口 =============
if __name__ == "__main__":
    visualize_ocr_results_with_translation(
        "OCR/images/image1.jpg",
        "ocr_output/test_result_image1.json",
        """
[1] 简短示例翻译
[2] 这是一个非常非常非常长的翻译文本，用来测试换行功能是否正常工作，
确保箭头指向真实的文本框而不是整个槽位
[3-4] 合并区块 -> 这是一个用于合并区域的长翻译文本
""",
        "output/vis_final.jpg"
    )
