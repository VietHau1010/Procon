# Giai đoạn 1: học giá trị trạng thái cuối ngày

Module thử nghiệm **độc lập**, không thay code chấm điểm của planner và không
thay model trong bất kỳ run/campaign nào đang có. Chỉ thư viện chuẩn Python.

## Mục tiêu

Không chỉ học “hôm nay được bao nhiêu udon”. Mỗi kế hoạch ứng viên được mô phỏng
đến cuối ngày, rồi **chơi tiếp mọi ngày còn lại** bằng một continuation policy cố
định. Nhãn là ba đại lượng riêng:

1. Loại mới thu thêm được **sau ngày hiện tại**.
2. Tổng số loại theo ngày thu được **trong những ngày còn lại**.
3. Tổng phần udon thu được **trong những ngày còn lại**.

Không gộp L/D/Q thành một scalar. `total_gain = gain_today + future`, nên điểm
hôm nay không bị đếm hai lần. Đây là kết quả của continuation policy với ngân
sách chỉ định, **không phải giá trị tối ưu thật**.

## Tạo dữ liệu

- Baseline đóng băng gồm luật, simulator, planner và map như pipeline train cũ.
- Đúng một xe tiếp, vị trí theo centrality của baseline; không đổi role giữa các
  nhánh. Đây là lựa chọn teacher rõ ràng, chưa phải tối ưu role.
- Mỗi map lấy tối đa `states-per-map` ngày phân bố đều trong trận. Mỗi trạng thái
  thử kế hoạch teacher, các seed tìm kiếm khác và bài idle làm negative example.
- Các nhánh chung trạng thái đầu ngày và hành động của đối thủ ngày đó. Những
  ngày sau được mô phỏng lại riêng cho từng nhánh; vị trí/xăng, brand đã lấy và
  traffic 2 ngày đều truyền tiếp, stock reset mỗi ngày.
- Ba đối thủ là baseline safe-solution với seed cố định. Không phải replay của
  đối thủ thực, không giả định biết hành động tương lai của họ khi trích feature.
- Chỉ lưu mẫu khi mọi bài hợp lệ theo `strict_fuel` và rollout hoàn tất đủ ngày.
  Trạng thái cuối trùng nhau (cả vị trí, xăng, brand, stay và serving) được gộp.
- Checkpoint từng map có checksum; collect chạy lại sẽ bỏ map đã hoàn tất. Map
  đang chạy khi Ctrl-C được thực hiện lại, không dùng nhãn dở dang.

## Đặc trưng và mô hình

27 feature không phụ thuộc ID ô: số ngày/bước còn lại, xăng mean/min/std, tỷ lệ xe
kẹt xăng, khoảng cách tới tanker, độ phân tán patrol, brand/serving miễn phí khi
đỗ ở quán, khoảng cách/tầm với brand chưa lấy, tiếp cận theo xăng và giao thông.
Chỉ dùng bản đồ được công bố, trạng thái kết thúc và lịch sử/dự báo giao thông.
**Không đọc nhãn rollout hoặc hành động đối thủ tương lai**.

Model khởi đầu là **ridge regression ba đầu ra**, không neural network:

- Chuẩn hóa input chỉ từ train.
- Nhãn chia theo cận lỏng của chính map/ngày để tránh map lớn lấn át map nhỏ.
- Chọn hệ số regularization bằng MSE validation trong grid cố định.
- Clamp output trong cận luật; ngày cuối có giá trị tương lai bằng 0.
- Test không được đọc trong lúc fit; chỉ mở bằng lệnh evaluate riêng.

## Chạy pipeline

Ở thư mục gốc repo, chọn run-dir mới:

```powershell
python -m hexudon.training.value init --run-dir artifacts/value-main --reference-run artifacts/training-demo-v3 --train-maps random8,random10 --train-seeds 30,31 --validation-maps random12 --validation-seeds 1030,1031 --test-maps random14 --test-seeds 2030,2031 --iterations 15 --rollout-iterations 5 --candidates 4 --states-per-map 3
python -m hexudon.training.value collect --run-dir artifacts/value-main --split train --jobs 2
python -m hexudon.training.value collect --run-dir artifacts/value-main --split validation --jobs 2
python -m hexudon.training.value fit --run-dir artifacts/value-main
python -m hexudon.training.value collect --run-dir artifacts/value-main --split test --jobs 2
python -m hexudon.training.value evaluate --run-dir artifacts/value-main --split test
```

Không truyền `reference-run` thì teacher dùng trọng số mặc định planner hiện tại.
Reference có mã khác snapshot hiện hành bị từ chối; không lén thay baseline.

Sau khi pipeline chạy ổn, tạo run khác với 20–50 map random nhiều kích thước,
map sample/replay config, nhiều ngày ngắn và map thiếu xăng. Tăng dần lên
`--iterations 100 --rollout-iterations 50 --candidates 8 --states-per-map 4`.
Chi phí xấp xỉ: số trạng thái × số kế hoạch × số ngày còn lại × số bot.
Đừng tăng tất cả tham số cùng lúc hoặc vừa train vừa thi trên cùng laptop.

Dataset train/validation/test được chia theo **map**, không chia các trạng thái
của cùng một map sang hai split. Ngân sách, nguồn và hash được khóa trong run.
Không sửa code value trong lúc collect/fit; code đổi thì tạo run mới.

## Tệp và cách hiểu kết quả

```text
value-config.json          # baseline, teacher weights, feature/runtime version
samples/<split>/<map>.json # group theo trạng thái, điểm hôm nay và nhãn future
value-model.json           # model thử nghiệm; automatic_live_deployment=false
value-fit-report.json      # train/validation, regularization chọn từ validation
value-evaluations/test.json
```

- `normalized_mse`: sai số từng đầu ra so với nhãn rollout; so thêm predictor
  hằng lấy trung bình train trong `constant_baseline_mse`.
- `same_today_score_pairs`: cặp có cùng L/D/Q hôm nay nhưng kết quả còn lại khác;
  `same_today_pair_accuracy` mới đo trực tiếp được khả năng phân biệt tương lai.
  Nếu số cặp ít, metric không đủ đáng tin.
- `counterfactual_selection`: chọn một kế hoạch trong pool và nhìn nhãn rollout
  tương ứng so với kế hoạch baseline tại **cùng trạng thái**. Không phải thắng
  một trận chơi thực sự: lấy lựa chọn mới hôm nay sẽ làm đổi trạng thái/ngày sau.
- Cách xếp pool thử nghiệm cộng dự đoán future với điểm hiện tại, làm tròn tầng
  L/D, sau đó xét Q. Chưa đưa bộ chọn này vào client thi thật.

## Kết quả smoke test thực tế, chưa đạt cổng triển khai

Run `artifacts/value-rollout-20261120-v1`, dùng trọng số model `g00001-c002`:

| Split | Map | Trạng thái | Mẫu | Chọn tốt / hòa / kém baseline |
|---|---:|---:|---:|---:|
| Train | 4 | 12 | 45 | 5 / 5 / 2 |
| Validation | 2 | 6 | 20 | 0 / 5 / 1 |
| Test | 2 | 6 | 24 | 2 / 1 / 3 |

Test có 8 cặp cùng điểm hôm nay, pair accuracy 62,5%. MSE Q test khoảng 0,01654,
**kém** predictor hằng (0,00929). Không được tuyên bố model này vượt baseline.

## Cổng tiếp theo trước khi dùng model để chọn route

1. Thu thập thêm cặp cùng điểm hôm nay nhưng khác mức xăng, vị trí và lịch tanker.
2. Cải thiện dự đoán D/Q trên validation, không chỉ fit tốt trên train.
3. Đấu full-match giữa learned reranker và baseline: cùng pool route và cùng
   ngân sách giây, giữ phương án baseline làm fallback khi ngoài miền dữ liệu.
4. Test map chưa gặp, theo dõi mất loại và bài lỗi; chưa qua cổng thì giữ bot cũ.

Chưa triển khai bước 3–4 vào bot: đó là cố ý để không dùng model còn yếu trong
thi đấu. Việc thêm mạng neural/GNN là giai đoạn sau khi có đủ dữ liệu và baseline
ridge tốt để đối chiếu, không phải giải pháp tự động bảo đảm thắng cách biệt.
