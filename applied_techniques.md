# Tổng Hợp Các Kỹ Thế / Kỹ Thuật Đã Áp Dụng (Patches & VRAM Optimizations)

Tài liệu này tổng hợp chi tiết các lỗi hệ thống, kỹ thuật sửa lỗi (code patches) và các phương pháp tối ưu hóa bộ nhớ GPU (VRAM) đã được áp dụng để chạy thành công tiến trình huấn luyện mô hình Dynamics của **RISE** trên môi trường Kaggle GPU T4 (15GB VRAM).

---

## 1. Sửa Lỗi Lệch Kích Thước Tensor (Tensor Shape Mismatches)

### 1.1 Lỗi Lệch Chiều Sequence Length (`128` vs `act_len`)
* **Triệu chứng:** Lỗi crash mô hình trong forward pass:
  `RuntimeError: The size of tensor a (128) must match the size of tensor b (35) at non-singleton dimension 1`
* **Nguyên nhân:** Kích thước chuỗi văn bản đầu vào là 128. Tuy nhiên, chiều dài của chuỗi tokens hành động (`action_states`) ban đầu bị mã hóa cứng (hardcoded) là 26. Khi cấu hình `action_chunk` thay đổi (ví dụ: `action_chunk = 6`), chiều dài thực tế của tokens hành động là `7` (`action_chunk + 1` cho cls_token). Do công thức lặp lại dùng số cứng 26:
  `n_repeat = (seq_len + 26 - 1) // 26`
  kết quả lặp lại chỉ đạt 35 tokens (7 × 5), gây ra sự lệch pha kích thước `128 vs 35` khi cộng với `encoder_hidden_states`.
* **Giải pháp:** Chuyển sang tính toán động hoàn toàn dựa trên thuộc tính kích thước thực tế của tensor hành động:
  ```python
  # Tệp: models/ltx_models/action_encoder_control.py
  seq_len = encoder_hidden_states.shape[1]
  act_len = action_states.shape[1]  # Tính động (ví dụ: 7 hoặc 26)
  n_repeat = (seq_len + act_len - 1) // act_len
  action_states = action_states.repeat(1, n_repeat, 1)[:, :seq_len, :]
  encoder_attention_mask[:, :act_len] = True
  ```

### 1.2 Lỗi Lệch Chiều Số Khung Hình Latent VAE (`144` vs `192`)
* **Triệu chứng:** Lỗi lệch kích thước khi thực hiện phép tính trích xuất vector hoặc gộp các đặc trưng khung hình quá khứ và tương lai.
* **Nguyên nhân:** Công thức tính số khung hình ẩn (`latent_frames`) bị mã hóa cứng thông qua tỷ lệ giảm mẫu thời gian:
  `latent_frames = raw_frames // self.TEMPORAL_DOWN_RATIO + 1 + mem_size`
  Công thức tĩnh này không phản ánh chính xác kích thước thực tế sau khi gộp (concatenate) chiều động của các khung hình bộ nhớ và khung hình tương lai.
* **Giải pháp:** Định nghĩa lại `latent_frames` động sau khi gộp hai tensor VAE latents lại với nhau:
  ```python
  # Tệp: runner/finetune_trainer.py
  latents = torch.cat((mem_latents, future_video_latents), dim=2)
  latent_frames = latents.shape[2]  # Trích xuất động trực tiếp từ shape
  ```

---

## 2. Sửa Lỗi Tương Thích Thư Viện (Environment & Library Workarounds)

### 2.1 Lỗi Ghi Log `Tee.isatty` và các thuộc tính khác
* **Triệu chứng:** Crash do đối tượng ghi log tùy biến `Tee` (dùng để đồng thời ghi ra stdout và file log) bị thiếu một số thuộc tính chuẩn của luồng hệ thống mà thư viện bên ngoài gọi tới.
* **Giải pháp:** Bổ sung phương thức `__getattr__` để tự động ủy quyền (delegate) tất cả các yêu cầu thuộc tính chưa định nghĩa sang luồng gốc `sys.stdout`:
  ```python
  # Tệp: utils/__init__.py
  class Tee:
      def __init__(self, *files):
          self.files = files
      # ...
      def __getattr__(self, name):
          return getattr(self.files[0], name)
  ```

### 2.2 Lỗi Thiếu Biến `FLAX_WEIGHTS_NAME`
* **Triệu chứng:** `ImportError` phát sinh từ thư viện `transformers` do không tìm thấy hằng số `FLAX_WEIGHTS_NAME`.
* **Giải pháp:** Tự động kiểm tra và chèn hằng số này vào bộ thuộc tính của thư viện nếu nó bị thiếu:
  ```python
  # Tệp: utils/__init__.py
  import transformers.utils
  if not hasattr(transformers.utils, "FLAX_WEIGHTS_NAME"):
      transformers.utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
  ```

---

## 3. Tối Ưu Hóa Bộ Nhớ GPU (VRAM Optimizations)

Môi trường Kaggle GPU T4 có giới hạn VRAM rất khắt khe (~15GB). Do đó, các kỹ thuật tối ưu hóa sau đã được áp dụng:

### 3.1 Chuyển Text Encoder Sang CPU (CPU Offloading)
* **Chi tiết:** Bộ mã hóa văn bản T5-XXL đi kèm LTX-Video có kích thước khổng lồ (**4.7 Tỷ tham số**), tiêu tốn khoảng **~9.4 GB VRAM** ở định dạng `bfloat16` chỉ để lưu trọng số. Nếu để trên GPU, VRAM còn lại sẽ không đủ để thực hiện quá trình lan truyền ngược (backward pass) của mô hình chính.
* **Giải pháp:** 
  1. Đưa `self.text_encoder` lên **CPU** (`to("cpu")`), tận dụng tài nguyên RAM hệ thống lớn của Kaggle (30GB).
  2. Toàn bộ quá trình mã hóa văn bản được thực hiện trên CPU, các embedding kết quả sau đó được đẩy lên GPU (`.to(device)`) để huấn luyện.
  3. Kỹ thuật này giúp **tiết kiệm ngay lập tức ~9.4 GB VRAM GPU** mà không ảnh hưởng tới chất lượng huấn luyện.

### 3.2 Tối Ưu Hóa Batch Size và Gradient Accumulation
* **Chi tiết:** Thay vì dùng `batch_size: 2` và `gradient_accumulation_steps: 4`, cấu hình đã được điều chỉnh thành:
  * `batch_size: 1` (giảm tải VRAM tức thời xuống còn 1 nửa).
  * `gradient_accumulation_steps: 8` (tích lũy gradient qua 8 bước trước khi cập nhật).
* **Kết quả:** Động học huấn luyện được bảo toàn nguyên vẹn với kích thước batch hiệu dụng không đổi (**effective batch size = 8 = 1 * 8**), trong khi đỉnh VRAM tiêu thụ giảm đi đáng kể.

### 3.3 Tắt Color Jitter (Color Augmentation)
* **Chi tiết:** Phép biến đổi tăng cường màu sắc (`color_jitter`) trong tập dữ liệu sinh ra các phép toán nhân ma trận lớn (einsum) trên không gian màu HSV/RGB, tạo ra các tensor trung gian lớn trên GPU gây OOM.
* **Giải pháp:** Vô hiệu hóa tính năng này bằng cách đặt `use_color_jitter: false` trong cấu hình YAML.

### 3.4 Cấu Hình Memory Allocator
* **Chi tiết:** Đặt cấu hình phân bổ bộ nhớ của PyTorch tránh phân mảnh:
  `export PYTORCH_ALLOC_CONF=expandable_segments:True`
  giúp PyTorch quản lý hiệu quả hơn các phân đoạn bộ nhớ được giải phóng trên GPU T4.

### 3.5 Bộ đệm ẩn văn bản (Prompt Embedding Caching)
* **Chi tiết:** Khi chuyển Text Encoder sang CPU, mặc dù giải phóng được 9.4GB VRAM nhưng suy luận (inference) mô hình 4.7 tỷ tham số trên CPU của Kaggle cực kỳ chậm (mất khoảng ~5-8 giây cho mỗi lượt forward pass). Với 8 bước tích lũy gradient (`gradient_accumulation_steps: 8`), mỗi bước huấn luyện thực tế mất khoảng **70.77 giây** -> Huấn luyện 600 bước sẽ mất gần **12 tiếng** (đạt ngưỡng giới hạn thời gian chạy của Kaggle và không thể chạy tiếp các bài thực nghiệm Benchmark khác).
* **Giải pháp:** 
  1. Do các tác vụ trong robot datasets (`svla`, `aloha`) chỉ chứa một lượng rất ít các câu lệnh mô tả nhiệm vụ (captions) cố định lặp đi lặp lại qua các epoch, ta tiến hành lưu đệm ẩn (caching) các prompt embeddings đã tính toán thông qua một dictionary `self.prompt_cache`.
  2. Ở bước huấn luyện tiếp theo, nếu câu lệnh đã được tính toán trước đó, hệ thống sẽ lấy trực tiếp từ bộ đệm (mất **0.0 giây**) thay vì phải gọi lại mô hình T5 trên CPU.
* **Kết quả:** Tốc độ huấn luyện tăng vọt **gấp 6 - 7 lần** (thời gian mỗi bước giảm từ 70.77s xuống còn **~8 - 12s**). Tổng thời gian train 600 bước giảm xuống chỉ còn **~1.5 giờ**, hoàn toàn dư dả thời gian để chạy đầy đủ Evaluation và 3 cấu hình của Benchmark trong cùng 1 phiên làm việc của Kaggle.
