# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026 

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** Python 3.13.15 / PyTorch 2.11.0+cpu / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
    - class_id: 468
    - class_name: "cab"
    - rank: 1
    - score: 0.510915
    - taxonomy_name: "ImageNet-1K"
- Record này mô tả toàn ảnh như thế nào? 
    - Record này gán một nhãn phân loại duy nhất (`cab` - xe taxi) đại diện cho toàn bộ nội dung bức ảnh ở cấp độ toàn cục (image-level). Nó chỉ ra chủ thể/ngữ cảnh nổi bật nhất mà hoàn toàn không cung cấp thông tin vị trí không gian, đồng thời bỏ qua các đối tượng khác cùng xuất hiện trong khung cảnh giao thông.
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
    - Checkpoint được huấn luyện và tối ưu hóa dựa trên một bộ dữ liệu lớn đã được gán nhãn trước (ví dụ: ImageNet, COCO).
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
    - **`ID` (`class_id`)**: Giúp máy tính và mã nguồn xử lý nhanh chóng, chính xác, làm chỉ mục (index) mảng/tensor mà không lo lỗi chính tả hay bảng mã văn bản.
    - **Tên lớp (`class_name`)**: Giúp con người (annotator, reviewer, developer) đọc hiểu trực tiếp ý nghĩa của nhãn.
    - **Tên taxonomy (`taxonomy_name`)**: Định danh hệ quy chiếu nhãn (ví dụ: `ImageNet-1K`, `COCO-80`). Điều này tránh xung đột dữ liệu, vì cùng một `class_id` ở các taxonomy khác nhau có thể biểu diễn các đối tượng hoàn toàn khác nhau (ví dụ: ID 0 ở COCO là `person`, nhưng ở ImageNet là loài cá `tench`).
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
    - **Quy tắc chọn chủ thể chính (Dominant Subject)**: Hướng dẫn ưu tiên chọn vật thể chiếm diện tích lớn nhất, nằm ở tiền cảnh/trung tâm ảnh (foreground/center), hoặc rõ nét nhất.
    - **Quy tắc ngữ cảnh chung (Scene-level)**: Quy định khi không có chủ thể vượt trội thì chọn nhãn cảnh quan chung hoặc chuyển sang tác vụ multi-label/object detection.
    - **Quy trình xử lý ngoại lệ (Tie-breaking & Escalation)**: Quy định cách chọn khi các đối tượng ngang hàng hoặc cách chuyển câu hỏi lên reviewer/team lead để giải quyết.
- Vì sao model score không phải ground truth?
    - **Model score** chỉ là độ tin cậy/xác suất ước lượng của thuật toán qua hàm Softmax dựa trên tri thức học được từ dữ liệu huấn luyện. Mô hình có thể bị "tự tin sai" (overconfident: điểm rất cao nhưng đoán sai nhãn do ảnh lạ, góc khuất, hoặc nhiễu).
    - **Ground truth** là "sự thật nền" chuẩn xác do con người (annotator, chuyên gia, reviewer) xác minh và thống nhất theo guideline, được dùng làm tiêu chuẩn đánh giá độ chính xác của mô hình.


## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
    - `class_name`: "person"
    - `score`: 0.912625
    - `bbox_xyxy`: [385.33, 69.24, 498.92, 348.92]
    - `bbox_width`: 113.58
    - `bbox_height`: 279.68
- Diễn giải vị trí box bằng lời:
    - Hộp viền bao quanh người đầu bếp mặc áo trắng, đeo tạp dề trắng và dây đeo quần đang đứng quay lưng lại camera ở vị trí trung tâm lệch sang phải của căn bếp. Box có kích thước khoảng 113.58 x 279.68 pixel, kéo dài từ ngang tầm vai/đầu (y ≈ 69) xuống ngang bắp chân (y ≈ 349).
- So sánh số prediction ở hai threshold:
    - Ở ngưỡng cao (`threshold = 0.60`), mô hình chỉ phát hiện **6 vật thể** có độ tự tin cao nhất (2 person, 2 oven, 2 bowl).
    - Ở ngưỡng chuẩn (`threshold = 0.35`), số lượng dự đoán tăng lên **11 vật thể** (phát hiện thêm các vật thể nhỏ hơn gồm 3 bowl và 2 cup trên bàn). Khi giảm ngưỡng, số lượng vật thể dự đoán tăng lên gần gấp đôi.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
    - **Độ bao phủ (Coverage / Recall)**: Ngưỡng thấp (0.35) bao phủ được nhiều vật thể hơn, hạn chế bỏ sót các đối tượng nhỏ hoặc bị che khuất; ngưỡng cao (0.60) có độ bao phủ thấp, dễ bỏ sót đối tượng thực tế trong ảnh.
    - **Khối lượng việc của Reviewer**: Với ngưỡng thấp, reviewer phải duyệt số lượng box lớn hơn và tốn thời gian loại bỏ các dự đoán sai/nhiễu (False Positives). Với ngưỡng cao, reviewer duyệt ít box hơn nhưng phải tự tay vẽ bù (manual annotation) cho các vật thể bị bỏ sót.
- Đề xuất một quy tắc box chặt:
    - Bốn cạnh của bounding box (trên, dưới, trái, phải) phải bao khít sát đến từng pixel rìa ngoài cùng của vật thể; không để khoảng trống thừa thãi (padding/margin), không bao trùm bóng đổ (shadow), và không cắt lẹm vào bất kỳ bộ phận nào của đối tượng.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
    - **Bị che khuất (Occluded)**: Guideline cần quy định rõ vẽ box bao quanh phần nhìn thấy được (Visible box) hay vẽ ước lượng toàn bộ hình thể thực tế (Amodal box); đồng thời quy định tỷ lệ che khuất tối đa (ví dụ > 70% diện tích bị che thì bỏ qua hay vẫn gán nhãn).
    - **Bị cắt mép (Truncated)**: Guideline quy định cạnh của box phải dừng sát mép khung hình ảnh (tọa độ tại biên 0 hoặc W, H), không suy đoán vượt ngoài biên ảnh.
    - **Escalation**: Khi vật thể bị che khuất quá nặng dẫn đến mơ hồ về phân lớp (không rõ là bát hay cốc, người thật hay đồ vật), annotator cần gắn cờ `escalate` để Lead/Reviewer ra quyết định thống nhất.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
    - `instance_id`: "kitchen-001"
    - `class_name`: "person"
    - `score`: 0.899318
    - `polygon_point_count`: 348
    - `polygon_xy` (trích đoạn một số điểm đầu): [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0], [441.0, 73.0], [439.0, 73.0], [438.0, 74.0], [437.0, 74.0], [436.0, 75.0]]
- Polygon bổ sung chi tiết gì so với box?
    - **Độ chính xác cấp độ pixel (Pixel-level boundary)**: Bounding box chỉ là hình chữ nhật bao quanh nên chứa rất nhiều điểm ảnh thuộc nền (background) hoặc các vật thể khác lọt vào góc (khoảng trống giữa hai chân, vùng nách, khoảng trống quanh đầu).
    - **Mô tả hình dạng thực tế (Contour/Shape)**: Polygon gồm 348 điểm uốn lượn ôm sát theo đường biên giải phẫu cơ thể của người đầu bếp (lọn tóc, vai, cánh tay, tạp dề, chân), cho biết chính xác **diện tích thực**, **chu vi** và phân định từng pixel nào thuộc về đối tượng, pixel nào là nền bếp.
- `instance_id` dùng để làm gì và không phải loại ID nào?
    - **Dùng để làm gì**: Dùng để **định danh và phân biệt duy nhất từng cá thể riêng biệt (Individual Instance)** trong cùng một ảnh (ví dụ: `kitchen-001` là người đầu bếp, phân biệt với người khác hay vật thể khác). Giúp mô hình đếm chính xác số lượng từng cá thể và tách biệt các đối tượng cùng lớp khi đứng cạnh hoặc chồng đè lên nhau.
    - **Không phải loại ID nào**:
        - Không phải là `class_id` (mã định danh lớp chung, ví dụ lớp người là `0`).
        - Không phải là `sample_id` hay `coco_image_id` (mã định danh bức ảnh).
        - Không phải là Global Tracking ID / Re-ID (chỉ có giá trị cục bộ trong ảnh hiện tại, không dùng để theo dõi danh tính xuyên suốt các camera hay video khác nhau).
- Đề xuất một quy tắc biên mask:
    - Đường biên polygon phải **ôm khít sát ranh giới pixel thực tế** của vật thể với dung sai không quá 1 - 2 pixel:
        - **Không tràn viền (No background leakage)**: Không để viền polygon lấn sang phần nền (tường, kệ bếp).
        - **Không cắt lẹm (No erosion)**: Không cắt phạm làm mất các chi tiết rìa như quai tạp dề, ngón tay.
        - **Mật độ điểm đa giác vừa đủ**: Đủ số lượng đỉnh để mô tả mượt mà các đường cong tự nhiên, không vẽ gấp khúc thô kệch và không tạo điểm dư thừa.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
    - **Vùng tiếp xúc / chồng lấn (Occlusion & Touching)**: Guideline cần quy định rõ ranh giới thuộc về đối tượng nào (thường đối tượng tiền cảnh sẽ cắt đứt mask của đối tượng hậu cảnh, không để các mask chồng đè lên nhau).
    - **Vùng mờ, bóng đổ, tóc (Soft boundaries & Shadows)**: Guideline phải quy định rõ **loại bỏ bóng đổ (cast shadow)** khỏi mask; với vùng viền mờ hoặc sợi tóc, quy định lấy ranh giới ở mức chuyển đổi độ tương phản (threshold gradient 50%).
    - **Escalation**: Khi hai đối tượng dính liền hoặc vùng biên quá nhòe khiến annotator không thể xác định điểm kết thúc của vật thể này và điểm bắt đầu của vật thể kia, annotator phải gắn cờ `escalate` để Lead/Reviewer ra quyết định ranh giới chuẩn.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 nhãn cấp ảnh (`class_id`, `class_name` thuộc taxonomy ImageNet-1K). | Ảnh có nhiều chủ thể (ví dụ vừa có taxi, bus, người); mô hình nhầm lẫn ngữ cảnh (đoán chảo/nồi thành `gong`) hoặc tự tin sai (overconfident). | Đọc kỹ guideline về quy tắc chọn chủ thể chính (vật thể to nhất, ở trung tâm/foreground); gán đúng 1 nhãn đại diện; gắn cờ escalate nếu ảnh mơ hồ. | Kiểm tra xem nhãn được gán có đúng với chủ thể chính theo guideline không; rà soát các trường hợp ảnh có nhiều đối tượng hoặc bối cảnh phức tạp. |
| Phát hiện vật thể | Danh sách các bounding box pixel dạng `xyxy = [x_min, y_min, x_max, y_max]` kèm `class_id`, `class_name`. | Bỏ sót vật thể nhỏ/khuất khi threshold cao; sinh ra dự đoán rác/nhiễu khi threshold thấp (ví dụ nhận nhầm bàn tay thành người); box bị lỏng hoặc cắt lẹm. | Tìm và vẽ box khít sát (tight box) cả 4 cạnh cho từng đối tượng thuộc danh mục lớp; xử lý đối tượng bị che khuất hoặc cắt mép theo đúng guideline. | Soi từng bounding box xem có bỏ sót vật thể không (Recall), có box rác không (Precision), box có bao khít sát không (IoU), và nhãn lớp có chính xác không. |
| Instance segmentation | Tập hợp tọa độ các đỉnh đa giác `polygon_xy` (hoặc binary mask pixel) kèm `instance_id` và `class_id`, `class_name`. | Mặt nạ bị lem sang vùng nền (background leakage), cắt lẹm chi tiết mảnh (ngón tay, quai tạp dề); khó xác định ranh giới giữa các vật thể tiếp xúc hoặc chồng lấn. | Dùng công cụ vẽ đa giác (polygon) ôm khít sát đường viền pixel thực tế của từng cá thể riêng biệt; tách biệt bằng `instance_id`; loại bỏ bóng đổ. | Phóng to (zoom) kiểm tra chất lượng đường viền mask ở cấp độ pixel; kiểm tra mask có bị lấn nền, mất chi tiết hay dính chùm các cá thể gần nhau không; yêu cầu rework nếu cẩu thả. |
 
## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
    - Tuyệt đối không sao chép, tải về máy cá nhân trái phép hoặc phát tán hình ảnh, dữ liệu huấn luyện nội bộ ra bên ngoài; không đưa bất kỳ thông tin định danh cá nhân nào (PII như họ tên, MSSV, CCCD, số điện thoại, email) vào báo cáo, code hoặc tập dữ liệu công khai; tuân thủ đầy đủ điều khoản bản quyền và giấy phép của dữ liệu (License).
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
    - Giảng viên hướng dẫn / Team Lead / Quản lý dự án (Project Manager / Data Security Officer) để được hướng dẫn xử lý và thu hồi dữ liệu không phù hợp theo đúng quy trình.



## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
