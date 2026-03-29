import os
import nbformat as nbf

with open('../Resources/lab07_text.txt', 'r', encoding='utf-8') as f:
    full_text = f.read()

def extract_between(start_str, end_str):
    start_idx = full_text.find(start_str)
    if end_str:
        end_idx = full_text.find(end_str, start_idx)
        return full_text[start_idx:end_idx].strip()
    return full_text[start_idx:].strip()

nb = nbf.v4.new_notebook()

# INTRO
text_1 = extract_between("1.Mục tiêu bài học", "3.Các phương thức dùng chung")
nb.cells.append(nbf.v4.new_markdown_cell("# BÀI THỰC HÀNH 07: EDGE-BASED SEGMENTATION & ACTIVE CONTOUR\n\n" + text_1))
text_3 = extract_between("3.Các phương thức dùng chung", "3.1. Đọc ảnh xám")
nb.cells.append(nbf.v4.new_markdown_cell(text_3))
nb.cells.append(nbf.v4.new_code_cell('''import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import active_contour
from skimage.filters import gaussian
import warnings
warnings.filterwarnings('ignore')

def read_gray_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError(f"Khong doc duoc anh: {path}")
    return img

def ensure_uint8(img):
    img = np.asarray(img)
    if img.dtype == np.uint8: return img
    return np.clip(img, 0, 255).astype(np.uint8)

def normalize_to_uint8(img):
    img = np.asarray(img).astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8: return np.zeros_like(img, dtype=np.uint8)
    return ((img - mn) / (mx - mn) * 255.0).astype(np.uint8)

def show_images(images, titles=None, cols=3, figsize=(15, 8), cmap="gray"):
    n = len(images)
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        if img.ndim == 3: plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else: plt.imshow(img, cmap=cmap)
        if titles: plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def draw_contours_on_image(img_gray, contours, color=(0, 255, 0), thickness=2):
    base = cv2.cvtColor(ensure_uint8(img_gray), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(base, contours, -1, color, thickness)
    return base

def overlay_mask_on_image(img_gray, mask, alpha=0.4):
    base = cv2.cvtColor(ensure_uint8(img_gray), cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    overlay[mask > 0] = (255, 0, 0)
    return cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)

def contours_to_mask(img_shape, contours):
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, thickness=-1)
    return mask

def snake_to_mask(img_shape, snake):
    mask = np.zeros(img_shape, dtype=np.uint8)
    pts = np.round(snake[:, ::-1]).astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask
'''))

# SECTION 4
text_4_intro = extract_between("4.Contour Maps and Edge Representations", "4.3. Code hướng dẫn")
nb.cells.append(nbf.v4.new_markdown_cell(text_4_intro))
nb.cells.append(nbf.v4.new_code_cell('''def compute_gradient_magnitude(img_gray):
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    return normalize_to_uint8(np.sqrt(gx**2 + gy**2))

def gradient_threshold_edge_map(grad, thresh=50):
    edge = np.zeros_like(grad, dtype=np.uint8)
    edge[grad >= thresh] = 255
    return edge

def canny_edge_map(img_gray, low=80, high=160, blur_kernel=5):
    blur = cv2.GaussianBlur(img_gray, (blur_kernel, blur_kernel), 1.0)
    return blur, cv2.Canny(blur, low, high)
    
def extract_external_contours(edge_map):
    contours, hierarchy = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy
'''))
text_4_ex = extract_between("4.4. Gợi ý loại ảnh mẫu", "5.Edge Linking and Gap Closing")
nb.cells.append(nbf.v4.new_markdown_cell(text_4_ex))

nb.cells.append(nbf.v4.new_markdown_cell(r'''### Bài 1: So sánh các biểu diễn biên'''))
nb.cells.append(nbf.v4.new_code_cell('''img_4_1 = read_gray_image('../Resources/coins1.jpg')
grad_4_1 = compute_gradient_magnitude(img_4_1)
edge_4_1 = gradient_threshold_edge_map(grad_4_1, thresh=100)
_, canny_4_1 = canny_edge_map(img_4_1, 50, 150)
show_images([img_4_1, grad_4_1, edge_4_1, canny_4_1],
            ['Anh goc', 'Gradient Magnitude', 'Threshold Edge', 'Canny Edge Map'], cols=4, figsize=(20, 5))
print("Nhận xét: Gradient Magnitude thể hiện độ mạnh yếu khác nhau của viền (mức xám đa dạng). Edge Map khi dùng ngưỡng đã loại được một số nét mờ, nhưng nét viền lại rất dày. Canny edge trả về nét mảnh chuẩn xác 1 px.")'''))

nb.cells.append(nbf.v4.new_markdown_cell(r'''### Bài 2: Vẽ contour lên ảnh gốc'''))
nb.cells.append(nbf.v4.new_code_cell('''contours_4_2, _ = extract_external_contours(canny_4_1)
ol_4_2 = draw_contours_on_image(img_4_1, contours_4_2, color=(255,0,0))
show_images([img_4_1, ol_4_2], ['Anh goc', 'Contour Overlay'], cols=2, figsize=(10, 5))'''))


# SECTION 5
text_5_intro = extract_between("5.Edge Linking and Gap Closing", "5.3. Code hướng dẫn")
nb.cells.append(nbf.v4.new_markdown_cell(text_5_intro))
nb.cells.append(nbf.v4.new_code_cell('''def edge_linking_with_closing(edge_map, kernel_size=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, kernel)

def filter_small_edge_components(binary_img, min_area=30):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    clean = np.zeros_like(binary_img)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area: clean[labels == i] = 255
    return clean
'''))
text_5_ex = extract_between("5.4. Gợi ý loại ảnh mẫu", "6.From Edge Maps to Closed Boundaries")
nb.cells.append(nbf.v4.new_markdown_cell(text_5_ex))

nb.cells.append(nbf.v4.new_markdown_cell(r'''### Bài 1: So sánh trước và sau edge linking'''))
nb.cells.append(nbf.v4.new_code_cell('''img_5_1 = read_gray_image('../Resources/leaves (1).png')
_, canny_5_1 = canny_edge_map(img_5_1, 50, 100, blur_kernel=3)
linked_5_1 = edge_linking_with_closing(canny_5_1, 5)

show_images([canny_5_1, linked_5_1], ["Edge map trưóc linking (Bị hở đứt)", "Linked Edge map sau closing (Hàn liền)"], cols=2, figsize=(12, 6))
print("Nhận xét: Morphological closing giúp các phân mảnh hở nối liền.")'''))

nb.cells.append(nbf.v4.new_markdown_cell(r'''### Bài 2: Ảnh hưởng của kích thước kernel'''))
nb.cells.append(nbf.v4.new_code_cell('''img_5_2 = read_gray_image('../Resources/coins2.jpg')
_, canny_5_2 = canny_edge_map(img_5_2, 80, 150)

linked_3 = edge_linking_with_closing(canny_5_2, kernel_size=3)
linked_5 = edge_linking_with_closing(canny_5_2, kernel_size=5)
linked_7 = edge_linking_with_closing(canny_5_2, kernel_size=7)

cts_3, _ = extract_external_contours(linked_3)
cts_5, _ = extract_external_contours(linked_5)
cts_7, _ = extract_external_contours(linked_7)

show_images([linked_3, linked_5, linked_7, 
             draw_contours_on_image(img_5_2, cts_3), draw_contours_on_image(img_5_2, cts_5), draw_contours_on_image(img_5_2, cts_7)], 
            ["Linked Map (3x3)", "Linked (5x5)", "Linked (7x7)", 
             "Contour Overlay (3x3)", "Contour Overlay (5x5)", "Contour Overlay (7x7)"], cols=3, figsize=(15, 10))'''))


# SECTION 6
text_6_intro = extract_between("6.From Edge Maps to Closed Boundaries", "6.3. Code hướng dẫn")
nb.cells.append(nbf.v4.new_markdown_cell(text_6_intro))
nb.cells.append(nbf.v4.new_code_cell('''def edge_to_closed_boundary_pipeline(img_gray, canny1=80, canny2=160, min_comp_area=30, min_contour_area=100, kernel_size=5):
    blur = cv2.GaussianBlur(img_gray, (5, 5), 1.0)
    edges = cv2.Canny(blur, canny1, canny2)
    linked = edge_linking_with_closing(edges, kernel_size=kernel_size)
    clean = filter_small_edge_components(linked, min_area=min_comp_area)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    good_contours = [c for c in contours if cv2.contourArea(c) >= min_contour_area]
    return {"edges": edges, "linked": linked, "clean_edges": clean, "contours": good_contours, "overlay": draw_contours_on_image(img_gray, good_contours)}
'''))
text_6_ex = extract_between("6.4. Gợi ý loại ảnh mẫu", "7.Closed Contours and Region Filling")
nb.cells.append(nbf.v4.new_markdown_cell(text_6_ex))

nb.cells.append(nbf.v4.new_markdown_cell(r'''### Bài 1: Pipeline có và không có gap closing'''))
nb.cells.append(nbf.v4.new_code_cell('''img_6_1 = read_gray_image('../Resources/seed1.png')
out_no_gap = edge_to_closed_boundary_pipeline(img_6_1, 50, 100, kernel_size=1) 
out_with_gap = edge_to_closed_boundary_pipeline(img_6_1, 50, 100, kernel_size=9)

show_images([out_no_gap['overlay'], out_with_gap['overlay']], ["Contour Overlay (Không Gap Closing)", "Contour Overlay (Có Gap Closing K=9)"], cols=2, figsize=(15, 6))'''))

nb.cells.append(nbf.v4.new_markdown_cell(r'''### Bài 2: Lọc contour hợp lệ'''))
nb.cells.append(nbf.v4.new_code_cell('''out_area_10 = edge_to_closed_boundary_pipeline(img_6_1, 50, 100, min_contour_area=10)
out_area_500 = edge_to_closed_boundary_pipeline(img_6_1, 50, 100, min_contour_area=500)

show_images([out_area_10['overlay'], out_area_500['overlay']], ["Contour còn lại lọc area=10", "Contour còn lại lọc area=500"], cols=2, figsize=(15, 6))'''))


# SECTION 7
text_7_intro = extract_between("7.Closed Contours and Region Filling", "7.3. Code hướng dẫn")
nb.cells.append(nbf.v4.new_markdown_cell(text_7_intro))
nb.cells.append(nbf.v4.new_code_cell('''def contour_circularity(contour):
    area = cv2.contourArea(contour)
    perim = cv2.arcLength(contour, True)
    if perim < 1e-8: return 0.0
    return 4.0 * np.pi * area / (perim ** 2)

def filter_circular_contours(contours, min_area=100, min_circularity=0.75):
    return [c for c in contours if cv2.contourArea(c) >= min_area and contour_circularity(c) >= min_circularity]
'''))
text_7_ex = extract_between("7.4. Gợi ý loại ảnh mẫu", "8.Active Contour (Snake) Segmentation")
nb.cells.append(nbf.v4.new_markdown_cell(text_7_ex))

nb.cells.append(nbf.v4.new_markdown_cell(r'''### Bài 1: Lọc contour theo diện tích và chu vi'''))
nb.cells.append(nbf.v4.new_code_cell('''img_7_1 = read_gray_image('../Resources/leaves2.png')
out_7_1 = edge_to_closed_boundary_pipeline(img_7_1, kernel_size=5, min_contour_area=0)
all_contours = out_7_1['contours']

def filter_area_perim(contours, min_area, min_perim):
    return [c for c in contours if cv2.contourArea(c) >= min_area and cv2.arcLength(c, True) >= min_perim]

good_ct = filter_area_perim(all_contours, min_area=600, min_perim=150)
mask_7_1 = contours_to_mask(img_7_1.shape, good_ct)

print(f"Số lượng contour ban đầu: {len(all_contours)}")
print(f"Số lượng contour sau khi lọc theo DT+ChuVi: {len(good_ct)}")
show_images([draw_contours_on_image(img_7_1, all_contours), draw_contours_on_image(img_7_1, good_ct), mask_7_1], 
            ['Tat ca contour tren anh goc', 'Cac contour sau loc', 'Mask chứa Object hop le'], cols=3, figsize=(15, 5))'''))

nb.cells.append(nbf.v4.new_markdown_cell(r'''### Bài 2: Chỉ giữ vật thể gần tròn'''))
nb.cells.append(nbf.v4.new_code_cell('''img_7_2 = read_gray_image('../Resources/coins3.png')
out_7_2 = edge_to_closed_boundary_pipeline(img_7_2, 50, 100, kernel_size=7)

circular_ct = filter_circular_contours(out_7_2['contours'], min_area=300, min_circularity=0.6)
mask_7_2 = contours_to_mask(img_7_2.shape, circular_ct)

show_images([img_7_2, mask_7_2, overlay_mask_on_image(img_7_2, mask_7_2)], 
            ['Ảnh Gốc (Mix Đồ tròn/dẹt)', 'Mask (Chỉ Vật Thể Tròn)', 'Ảnh Overlay Trên Ảnh Gốc'], cols=3, figsize=(15, 5))'''))


# SECTION 8
text_8_intro = extract_between("8.Active Contour (Snake) Segmentation", "8.3. Code hướng dẫn")
nb.cells.append(nbf.v4.new_markdown_cell(text_8_intro))
nb.cells.append(nbf.v4.new_code_cell('''def run_active_contour(img_gray, init_contour, sigma=2.0, alpha=0.1, beta=0.2, gamma=0.01):
    img_float = img_gray.astype(np.float32) / 255.0
    img_smooth = gaussian(img_float, sigma=sigma)
    snake = active_contour(img_smooth, init_contour, alpha=alpha, beta=beta, gamma=gamma)
    return img_smooth, snake

def create_ellipse_init_contour(center_row, center_col, radius_row, radius_col, num_points=200):
    t = np.linspace(0, 2 * np.pi, num_points)
    return np.stack([center_row + radius_row * np.sin(t), center_col + radius_col * np.cos(t)], axis=1)

def draw_snake_on_image(img_gray, snake_points, color=(0, 255, 0), thickness=2):
    base = cv2.cvtColor(ensure_uint8(img_gray), cv2.COLOR_GRAY2BGR)
    pts = np.round(snake_points[:, ::-1]).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(base, [pts], isClosed=True, color=color, thickness=thickness)
    return base
'''))
text_8_ex = extract_between("8.4. Gợi ý loại ảnh mẫu", "9.Bài tập tổng hợp cho toàn bộ bài học")
nb.cells.append(nbf.v4.new_markdown_cell(text_8_ex))

nb.cells.append(nbf.v4.new_markdown_cell(r'''### Bài 1: So sánh initial contour tốt và xấu'''))
nb.cells.append(nbf.v4.new_code_cell('''img_8_1 = cv2.resize(read_gray_image('../Resources/cell1.jpg'), (256, 256))
init_good = create_ellipse_init_contour(130, 130, 80, 80)
init_bad = create_ellipse_init_contour(50, 50, 30, 30)

_, snake_good = run_active_contour(img_8_1, init_good, alpha=0.015, beta=10, gamma=0.001)
_, snake_bad = run_active_contour(img_8_1, init_bad, alpha=0.015, beta=10, gamma=0.001)

sg_mask = snake_to_mask(img_8_1.shape, snake_good)
sb_mask = snake_to_mask(img_8_1.shape, snake_bad)

show_images([draw_snake_on_image(img_8_1, snake_good), sg_mask, 
             draw_snake_on_image(img_8_1, snake_bad), sb_mask],
            ['Overlay (Good init)', 'Mask (Good Init)', 'Overlay (Bad Init)', 'Mask (Bad Init)'], cols=2, figsize=(10, 10))'''))

nb.cells.append(nbf.v4.new_markdown_cell(r'''### Bài 2: Ảnh hưởng của alpha, beta, gamma'''))
nb.cells.append(nbf.v4.new_code_cell('''_, snake_a1 = run_active_contour(img_8_1, init_good, alpha=0.5, beta=10, gamma=0.001)     # High alpha (Elasticity)
_, snake_b1 = run_active_contour(img_8_1, init_good, alpha=0.015, beta=100, gamma=0.001)  # High beta (Rigidity)
_, snake_g1 = run_active_contour(img_8_1, init_good, alpha=0.015, beta=10, gamma=0.1)     # High gamma (Edge Attraction)

show_images([draw_snake_on_image(img_8_1, snake_a1), draw_snake_on_image(img_8_1, snake_b1), draw_snake_on_image(img_8_1, snake_g1)],
            ['High Alpha (Căng rút lõm)', 'High Beta (Snake gồng mình cứng)', 'High Gamma (Lực hút nhạy biên, răng cưa)'], cols=3, figsize=(15, 5))'''))

nb.cells.append(nbf.v4.new_markdown_cell(r'''### Bài 3: So sánh contour filling và snake trên ảnh biên yếu'''))
nb.cells.append(nbf.v4.new_code_cell('''img_8_3 = cv2.resize(read_gray_image('../Resources/WBC2.jpg'), (256, 256))
out_8_3 = edge_to_closed_boundary_pipeline(img_8_3, 30, 80, kernel_size=5)
mask_contour = contours_to_mask(img_8_3.shape, out_8_3['contours'])

snake_init_8_3 = create_ellipse_init_contour(130, 130, 90, 90)
_, snake_c = run_active_contour(img_8_3, snake_init_8_3, alpha=0.015, beta=5, gamma=0.005)
mask_snake = snake_to_mask(img_8_3.shape, snake_c)

show_images([mask_contour, overlay_mask_on_image(img_8_3, mask_contour), mask_snake, overlay_mask_on_image(img_8_3, mask_snake)], 
            ["Mask ket qua (Contour)", "Overlay (Contour)", "Mask ket qua (Snake)", "Overlay (Snake)"], cols=2, figsize=(15, 10))'''))


# SECTION 9
text_9 = extract_between("9.Bài tập tổng hợp cho toàn bộ bài học", "10.Kết luận")
nb.cells.append(nbf.v4.new_markdown_cell(text_9))

nb.cells.append(nbf.v4.new_markdown_cell(r'''### Bài 1: Pipeline contour-based segmentation hoàn chỉnh'''))
nb.cells.append(nbf.v4.new_code_cell('''img_9_1 = read_gray_image('../Resources/coins1.jpg')
out_9_1 = edge_to_closed_boundary_pipeline(img_9_1, 50, 150, min_comp_area=50, min_contour_area=200, kernel_size=9)
mask_9_1 = contours_to_mask(img_9_1.shape, out_9_1['contours'])
overlay_mask_9_1 = overlay_mask_on_image(img_9_1, mask_9_1)
edge_overlay = overlay_mask_on_image(img_9_1, out_9_1['edges'])

show_images([edge_overlay, out_9_1['overlay'], mask_9_1, overlay_mask_9_1],
            ['Edge Overlay', 'Contour Overlay', 'Segmentation Mask', 'Mask Overlay'], cols=4, figsize=(20, 5))'''))

nb.cells.append(nbf.v4.new_markdown_cell(r'''### Bài 2: Nghiên cứu ảnh hưởng của gap closing'''))
nb.cells.append(nbf.v4.new_code_cell('''out_k3 = edge_to_closed_boundary_pipeline(img_9_1, 50, 150, kernel_size=3)
out_k9 = edge_to_closed_boundary_pipeline(img_9_1, 50, 150, kernel_size=9)

show_images([out_k3['edges'], out_k3['linked'], out_k3['overlay'], contours_to_mask(img_9_1.shape, out_k3['contours'])], 
            ["Edge Map (K=3)", "Linked Edges (K=3)", "Contour Overlay (K=3)", "Mask (K=3)"], cols=4, figsize=(20, 5))
            
show_images([out_k9['edges'], out_k9['linked'], out_k9['overlay'], contours_to_mask(img_9_1.shape, out_k9['contours'])], 
            ["Edge Map (K=9)", "Linked Edges (K=9) - Better", "Contour Overlay (K=9)", "Mask (K=9)"], cols=4, figsize=(20, 5))'''))

nb.cells.append(nbf.v4.new_markdown_cell(r'''### Bài 3: Shape-aware contour selection'''))
nb.cells.append(nbf.v4.new_code_cell('''img_9_3 = read_gray_image('../Resources/coins3.png')
out_9_3 = edge_to_closed_boundary_pipeline(img_9_3, 40, 120, min_contour_area=200, kernel_size=5)

# Selection: Chỉ giữ Object hình Thon dài hoặc không phải Tròn Đồng Xu.
non_circular = [c for c in out_9_3['contours'] if contour_circularity(c) < 0.6 and cv2.contourArea(c) > 500]
final_mask_9_3 = contours_to_mask(img_9_3.shape, non_circular)

show_images([img_9_3, final_mask_9_3], ['Ảnh Gốc Nhỏ + To', 'Mask: Chỉ chứa object Thon dẹt mong muốn'], cols=2, figsize=(15, 6))'''))

nb.cells.append(nbf.v4.new_markdown_cell(r'''### Bài 4: Snake trên ảnh biên yếu'''))
nb.cells.append(nbf.v4.new_code_cell('''img_9_4 = cv2.resize(read_gray_image('../Resources/WBC1.jpg'), (256, 256))
init_a = create_ellipse_init_contour(130, 130, 60, 60)
init_b = create_ellipse_init_contour(130, 130, 120, 120)

_, s1 = run_active_contour(img_9_4, init_a, alpha=0.01, beta=0.5, gamma=0.005)
_, s2 = run_active_contour(img_9_4, init_b, alpha=0.01, beta=0.5, gamma=0.005)

mask_s1 = snake_to_mask(img_9_4.shape, s1)
mask_s2 = snake_to_mask(img_9_4.shape, s2)

show_images([draw_snake_on_image(img_9_4, s1), mask_s1, draw_snake_on_image(img_9_4, s2), mask_s2],
            ['Snake S1 Overlay', 'Mask S1', 'Snake S2 Overlay', 'Mask S2'], cols=4, figsize=(20, 5))
print("Sự khác biệt: Nếu ban đầu tạo vòng nhỏ (S1), snake nở ra có thể bắt đúng nhân Tế Bào, nhưng nếu làm quá to (S2) bao hết thì Rắn lại bám vào màng rìa ngoài cùng mờ nhạt do lực hút phân tán.")'''))

nb.cells.append(nbf.v4.new_markdown_cell(r'''### Bài 5: Case study nhỏ trên một tập ảnh (Ảnh Hạt/Lá/Tiền)'''))
nb.cells.append(nbf.v4.new_code_cell('''test_paths = ['../Resources/seed1.png', '../Resources/leaves (1).png', '../Resources/coins1.jpg']
for p in test_paths:
    img = read_gray_image(p)
    out = edge_to_closed_boundary_pipeline(img, 40, 120, kernel_size=5)
    
    # Run active contour (Snake)
    img_rs = cv2.resize(img, (256, 256))
    init_snake = create_ellipse_init_contour(128, 128, 80, 80)
    _, snake_final = run_active_contour(img_rs, init_snake, alpha=0.02, beta=10, gamma=0.005)
    
    show_images([img, out['overlay'], contours_to_mask(img.shape, out['contours']), draw_snake_on_image(img_rs, snake_final)], 
                [f'Ảnh: {os.path.basename(p)}', 'Contour Overlay', 'Contour Mask', 'Snake Approach'], cols=4, figsize=(20, 4))
print("Nhận xét: Snake phù hợp ảnh có 1 object to giữa khung, còn Region filling Contour giải quyết gọn đa vật thể (coins).")'''))


text_10 = extract_between("10.Kết luận", "")
nb.cells.append(nbf.v4.new_markdown_cell(text_10))

with open('Lab07.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Lab07 successfully generated with ALL EXACT 16 EXERCISES mapped precisely to the document requirements.")
