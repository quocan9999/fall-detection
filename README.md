# Fall Detection Desktop App

Ứng dụng desktop Windows dùng PySide6 và checkpoint YOLO Pose đã fine-tune để nhận diện
`no_fall` / `fall` trên camera realtime, ảnh và video.

Model mặc định của ứng dụng là `weights/best.pt`, được lấy từ:

```text
D:\Home\Downloads\clean-no-resize-sleeping-as-fall\weights\best.pt
```

Checkpoint này là model đã train cho bài toán fall detection. Ứng dụng không
dùng file pretrained nền `yolo11m-pose.pt` để chạy nhận diện.

## Chức năng

- Camera realtime từ webcam laptop/PC, webcam USB hoặc Iriun Webcam.
- Vẽ bounding box, skeleton/keypoint, label và confidence cho kết quả YOLO Pose.
- Phát âm báo khi camera realtime phát hiện `fall`, giới hạn lặp lại mỗi 2 giây.
- Upload ảnh và video, xem preview annotate và lưu kết quả vào `outputs/`.
- Nạp một checkpoint `.pt` khác để thử model đã train ở phiên chạy hiện tại.
- Từ chối model không phải Pose hoặc không có đúng label `0=no_fall`, `1=fall`.
- Hiển thị lỗi khi không mở được camera/file/model, lỗi inference, hoặc không có
  đối tượng được nhận diện.

## Cài đặt và chạy

Yêu cầu: Windows và Python 3.10.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Ứng dụng mặc định sử dụng `CPU` để tương thích ổn định. Có thể chọn `auto` trong
giao diện; nếu backend tự động không chạy được, ứng dụng thử lại bằng CPU và
ghi thông báo. Giá trị inference mặc định là `960`, theo kích thước train của
checkpoint. Khi realtime chậm, chọn `640`, `480` hoặc `320`.

## Dùng camera điện thoại qua Iriun

1. Cài Iriun Webcam trên điện thoại và trên Windows.
2. Kết nối điện thoại và máy tính theo hướng dẫn của Iriun, kiểm tra video đã
   xuất hiện trong ứng dụng Iriun trên PC.
3. Mở ứng dụng fall detection, vào tab `Camera`, bấm `Quét camera`.
4. Chọn camera tương ứng với Iriun trong danh sách rồi bấm `Bắt đầu realtime`.

Iriun xuất camera điện thoại dưới dạng camera ảo Windows, vì vậy ứng dụng không
kết nối trực tiếp tới điện thoại qua mạng.

## Kết quả và model thay thế

- Ảnh annotate lưu tại `outputs/images/`.
- Video annotate lưu dạng MP4 tại `outputs/videos/`; video kết quả không giữ
  audio gốc.
- Nút `Nạp model .pt` chỉ đổi model trong phiên đang chạy. Model mới phải là
  YOLO Pose cho đúng hai lớp `no_fall` và `fall`; khi model lỗi hoặc sai contract,
  ứng dụng thông báo lỗi và tiếp tục giữ model hợp lệ trước đó.
- Không thể đổi model trong khi camera hoặc video/ảnh đang xử lý.

## Kiểm tra nhanh

```powershell
py -3.10 -m unittest discover -s tests -v
```

Các kiểm thử tập trung vào contract model và cách phân loại kết quả
`fall` / không nhận diện được. Kiểm thử camera và âm thanh cần chạy thủ công
trên thiết bị Windows thực tế.
