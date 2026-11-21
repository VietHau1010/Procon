# Huấn luyện chiến thuật HEXUDON offline

**Bước tiếp theo đã có:** [học giá trị cuối ngày bằng rollout nhiều ngày](value/README.md).
Đây là pipeline riêng `python -m hexudon.training.value`, chưa bật trong bot thi thật.

Đây là **học tiến hóa trọng số** quanh `planner_contest`, không phải deep RL hay
mạng neural. Mỗi ứng viên là một bộ trọng số, được chấm bằng việc chơi trọn trận
trong simulator. Không cần GPU, PyTorch, thư viện ngoài hay token.

## Cách hoạt động

1. `init` sao chép mã planner hiện tại, các phụ thuộc luật/mô phỏng và cấu hình map
   vào run. Mã local chưa commit cũng được lấy làm baseline. Hash phát hiện việc
   sửa baseline/dataset sau khi khởi tạo; không dùng baseline từ Git HEAD.
2. Tạo ứng viên bằng đột biến 1–3 trọng số: phần udon, giữ xăng, đỗ xe, mức khẩn
   cấp của loại còn thiếu, khả năng tiếp cận brand ngày sau và phạt chiếm đường.
   Hằng số luật và thứ tự L/D/Q không được học/thay đổi.
3. Baseline và ứng viên chơi **cùng trận**, cùng map/seed/ngân sách vòng ALNS,
   kèm hai bot baseline có ngân sách tìm kiếm 0 để tạo thêm giao thông. Mọi bài
   được kiểm bằng simulator đóng băng với `strict_fuel=True`.
4. Ứng viên chỉ thành champion nếu thắng ít nhất một ca train, không thua ca
   train/validation nào, không có bài lỗi. Khi đã có champion học được, phải
   qua cổng tương tự so với champion nữa. Không cộng tổng Q để che một trận mất L/D.
5. Lưu checkpoint sau **mỗi ứng viên hoàn tất**. Ctrl-C không nâng cấp ứng viên
   dở dang; chạy lại sẽ tiếp tục đúng trial chưa hoàn tất.

**Không thay mặc định của bot thi thật.** Model export là JSON, không phải mã
Python tự sinh. Chỉ trọng số và model trong thư mục run thay đổi khi học.

## Chạy nhanh để kiểm tra pipeline

Chạy ở gốc project; tên run phải chưa tồn tại:

```powershell
python -m hexudon train init --run-dir artifacts/train-quick --train-maps random10 --validation-maps random12 --test-maps random14 --train-seeds 0,1 --validation-seeds 1000 --test-seeds 2000 --iterations 20 --role-iterations 2 --population 4
python -m hexudon train run --run-dir artifacts/train-quick --generations 2 --jobs 2
python -m hexudon train status --run-dir artifacts/train-quick
python -m hexudon train evaluate --run-dir artifacts/train-quick --split test --jobs 2
```

`status` cho biết generation, số trial, số promotion, champion và trọng số hiện tại.
`evaluate` chỉ ghi báo cáo, **không** thay champion dù kết quả tốt hay xấu.

## Buổi train dài hơn

```powershell
python -m hexudon train init --run-dir artifacts/train-main --train-maps sample_16,random16,random24 --validation-maps random20,random28 --test-maps sample_24,sample_32,random8x32 --train-seeds 0,1,2,3 --validation-seeds 1000,1001 --test-seeds 2000,2001 --iterations 300 --role-iterations 40 --population 8
python -m hexudon train run --run-dir artifacts/train-main --generations 10 --jobs 2
```

Học liên tục cho tới khi bạn dừng:

```powershell
python -m hexudon train run --run-dir artifacts/train-main --generations 0 --jobs 2
```

Ctrl-C dừng tại checkpoint hoàn tất cuối cùng. Chạy lại cùng lệnh để tiếp tục.
`--generations 2` nghĩa là chạy thêm tới 2 thế hệ kể từ checkpoint hiện tại; mỗi
thế hệ có `population` ứng viên. Nếu dừng giữa thế hệ, phần còn lại được chạy trước.

- `--jobs` song song các ca map/seed, không làm thay đổi ngân sách vòng tìm kiếm.
  Nên bắt đầu 1–2 jobs, không vừa train vừa thi thật trên cùng laptop.
- Evaluator tiến hóa dùng một xe tiếp theo heuristic trung tâm cố định và tắt
  dense beam cuối ngày trong vòng khám phá để mỗi trial có thời gian hữu hạn.
  Đây chỉ là sàng lọc trọng số; model giữ lại phải được đánh giá bằng planner
  đầy đủ và kiểm tra online Mock Contest trước khi xem là cải thiện.
- Không tự bật train vô hạn trong nền; bạn chủ động chạy lệnh liên tục ở trên.
- Đúng **một xe tiếp nhiên liệu** được kiểm tra trong mọi lần đánh giá. Snapshot
  phải là bản planner hỗ trợ quy tắc này; nếu không, các ca sẽ bị đánh lỗi và
  không có promotion.
- `--players` là mẫu số giao thông (mặc định 4 đội). Giá trị >4 mô phỏng thêm
  đội đứng yên. Nó **không** tái tạo lưu lượng tất cả đối thủ trong replay thật.
- Map nhận tên mẫu, map random hoặc file cấu hình setup JSON, không phải file
  `-replay.json` chỉ chứa frame. Ví dụ `--train-maps artifacts/maps/2dd1ab59.json,random16`.

## Dữ liệu và chống học lệch

### Corpus nhiều tình huống từ map web và địa hình sinh mới

`hexudon.training.corpus` tạo tập dữ liệu lớn, tái lập được. Các map web đã biết
chỉ thuộc train/phát triển; validation và test lấy từ những họ địa hình sinh mới.
Mọi biến thể của cùng họ nằm trong **một** split, không được tính là những mẫu
test độc lập. Hash loại bản trùng, bỏ qua khác biệt timestamp/thời lượng mạng
và số đội nếu đó là khác biệt duy nhất. Kiểm tra đầu vào, stock và kết nối
không có nghĩa mọi brand đều thu được hay mọi bài planner đều hợp lệ.

```powershell
python -m hexudon.training.corpus build --web-maps artifacts/lookahead-mock-maps-20261120-v1 --out artifacts/scenario-corpus-20261120-v1 --random-families 24 --per-family 64 --seed 20261120
```

Bộ đầu tiên có **2.048 cấu hình duy nhất thuộc 32 họ**: 8 họ map Mock đã biết
và 24 địa hình sinh mới; train 1.280, validation 384, test 384. Gồm kho ít/nhiều/
không đều, quán tụ cụm/rải xa, ít/nhiều quán, brand hiếm/nhiều brand, 3–8 xe,
vị trí xuất phát, ít xăng, ngày ngắn và dài xen kẽ, chi phí địa hình và traffic.
Giữ kho đầu ngày và nhiên liệu khởi tạo theo luật hiện có, không tự tạo cơ chế
luật mới. Đây là cấu hình trận, chưa phải bộ phủ mọi trạng thái giữa trận.

Không đưa cả corpus vào một trial nhỏ rồi kết luận trainer bị treo. Khởi tạo
batch cân bằng giữa các họ, bắt đầu với map nhỏ; các index kế tiếp lấy ca mới
không lặp lại trong cùng tập ứng viên đủ điều kiện:

```powershell
python -m hexudon.training.corpus init-batch --corpus artifacts/scenario-corpus-20261120-v1 --run-dir artifacts/train-scenario-small-20261120-v1 --size 8 --index 0 --small --iterations 20 --population 2
python -m hexudon train run --run-dir artifacts/train-scenario-small-20261120-v1 --generations 1 --jobs 1
```

Mỗi batch là một run riêng được đóng băng, **không** tự gộp champion hay tự chuyển
trọng số giữa các run. Champion của batch chỉ qua cổng trên batch đó; phải kiểm
tra trên bộ phát triển rộng hơn, đủ planner/giây, và Mock trước khi triển khai.
`players=None` trong API khởi tạo giữ số đội của từng map (>=4 cho evaluator
bốn đội); không ghi đè mọi ca thành 4. Mẫu số nhiều đội vẫn không tái tạo đầy đủ
lưu lượng của đối thủ thật. CLI `train init` thông thường giữ mặc định 4 đội.
Không mở điểm test khi đang chọn thuật toán/trọng số; kiểm tra cấu hình test
không phải đánh giá kết quả test. Không ghi đè corpus hoặc run đã tồn tại.

- Train dùng để khám phá; validation là cổng chấp nhận được **dùng lặp lại**.
- Test không được đọc bởi vòng học; chạy đánh giá cuối bằng lệnh `evaluate`.
- Cùng map cấu hình không được nằm ở hai split, kể cả đổi giờ mở/số đội/thời gian
  phản hồi. Map mẫu với nhiều seed được gộp còn một ca; map random mới sinh theo seed.
- Nhiều lần điều chỉnh dựa trên validation có thể overfit validation. Nếu xem test
  rồi chỉnh train theo test, nó cũng không còn là test độc lập: cần tập test mới.
- Dataset được cố định để so sánh tái lập. Muốn bổ sung map mới, tạo run mới; không
  sửa JSON dataset của run đang học. Tiếp tục run cũ vẫn học thêm trọng số trên tập cũ.
- Nếu code planner của project thay đổi, run cũ vẫn dùng snapshot cũ. Nếu chính
  code training/evaluator/storage thay đổi, resume bị chặn: tạo run mới để không
  trộn hai quy trình đánh giá khác nhau.

## Kết quả được lưu

```text
artifacts/train-main/
  baseline/          # mã luật, simulator, planner/ và planner_contest/ đóng băng
  data/              # cấu hình đầy đủ train / validation / test
  manifest.json      # hash, ngân sách, không gian học, baseline weights
  baseline.json      # model baseline ban đầu
  state.json         # checkpoint có thẩm quyền
  champion.json      # model tốt nhất đã qua cổng; có thể vẫn là baseline
  models/            # các model được chấp nhận
  trials/            # trọng số thử + báo cáo từng ca, kể cả ứng viên bị loại
  evaluations/       # báo cáo evaluate độc lập
```

Một run chỉ có một writer, kể cả evaluate. Nếu máy bị tắt/kill để lại `.writer.lock`,
kiểm tra không còn tiến trình train/evaluate của run đó rồi mới xóa riêng file lock.
Không dùng lại một thư mục `init` dở dang; chọn tên run mới để tránh ghi đè dữ liệu.
Run nằm trong `artifacts/` đã được Git ignore: sao lưu riêng nếu cần chuyển máy.

## Dùng model một cách chủ động

```python
from hexudon.offline.mapgen import load_or_generate
from hexudon.training.inference import trained_strategist

cfg = load_or_generate("sample_16", seed=42)
bot = trained_strategist("artifacts/train-main", cfg,
                         search_iterations=300, role_iterations=40)
roles = bot.select_roles()
# bot.plan_day(DayState(...)) dùng mã và trọng số đã đóng băng của run.
```

`champion.json` không được tự động nạp bởi `web`, `play` hay `sandbox`. Trước khi
gắn model vào client thi thật, cần test thêm với ngân sách **giây**, độ trễ mạng,
thời gian chọn role và bộ map/seed rộng hơn.

## Giới hạn bảo đảm

Cổng promotion so **tuple L/D/Q**, không dùng thời gian tính trên máy làm tiêu chí
4 khi ngân sách là số vòng. `compute_seconds` được ghi để chẩn đoán. Đây không
phải cổng “không thua mọi trận theo thứ hạng chính thức”.

Không bảo đảm champion thắng trên mọi map chưa gặp, đạt 960 phần, hay điểm sẽ
tăng sau mỗi thế hệ. Khi không tìm thấy ứng viên đủ tốt, champion giữ nguyên.
Các tham số không thay cấu trúc route/decoder; muốn học policy neural hoặc toán
tử tìm kiếm mới sẽ là giai đoạn sau, cần dữ liệu và ngân sách đánh giá riêng.

### Smoke test đã chạy

`artifacts/training-demo-v3`: 2 thế hệ, 8 trial, 20 vòng/ngày, 2 vòng role, 2 jobs.
Một model được nhận: trên 2 map train L/D giữ 14/113, Q tăng 304 → 317; validation
không thua. Trên 1 map test riêng, hòa baseline 12/59/139. Tập này rất nhỏ, chỉ
chứng minh vòng học/resume/evaluate chạy được, không chứng minh mạnh hơn tổng quát.

## Campaign đối chiếu model hiện tại

Để tìm model thắng rõ một champion đã có, thay vì chỉ vượt trọng số mặc định:

```powershell
python -m hexudon.training.campaign --run-dir artifacts/convincing-campaign-20261120 --reference-run artifacts/training-demo-v3 --iterations 150 --role-iterations 10 --max-trials 128 --hours 2 --jobs 4
```

Campaign khóa champion của reference tại lúc khởi tạo. Không sửa reference run.
Mã planner của reference và mã hiện hành phải có cùng hash; nếu khác thì so chỉ
trọng số không còn công bằng và khởi tạo bị từ chối.

- Dữ liệu riêng: 6 cấu hình train, 6 validation, 24 test. Các map test random
  không phải những map smoke test đã biết.
- Thay đổi chỉ được giữ làm champion phát triển khi không thua reference/champion
  cũ trên train và validation. Đây chưa phải model đạt yêu cầu cuối cùng.
- Finalist phải thắng >=60% validation rồi mới mở test. Cổng cuối cần >=20 ca,
  thắng >=60%, không thua, không bài lỗi và p <=0.01 theo kiểm định dấu hai phía
  trên các ca không hòa. Với tập 24 ca mặc định, cần ít nhất 15 thắng.
- Nếu trượt test, lần xét finalist sau dùng map random mới và seed mới, không
  thử đi thử lại trên cùng tập test đã lộ kết quả. Kết quả kiểm định chỉ có ý
  nghĩa với giả định các ca thử đủ độc lập; không phải bảo đảm thắng mọi map.
- Cổng này đo L/D/Q ở cùng ngân sách vòng, **không** đo thứ hạng theo thời gian
  phản hồi khi thi thật. Cần kiểm tra riêng trước khi triển khai.
- Kết thúc khi đạt cổng hoặc hết `--max-trials`/`--hours`. Giới hạn giờ kiểm tra
  giữa các đánh giá hoàn tất, nên có thể vượt nhẹ để hoàn tất trận đang chạy.

Tệp `campaign-state.json` là checkpoint; `campaign-champion.json` là model phát
triển tốt nhất. **Chỉ khi có `qualified.json` và `qualified: true`** mới đạt cổng
cuối nói trên. `champion.json` thuộc pipeline train thông thường, không phải
champion của campaign. Model trong `qualified.json["model"]` có thể được xuất
thành JSON và nạp bằng `trained_strategist` sau bước kiểm tra thời gian thực.

Tiếp tục một đợt mới từ checkpoint (không đổi dataset/baseline đã khóa):

```powershell
python -m hexudon.training.campaign --run-dir artifacts/convincing-campaign-20261120 --max-trials 128 --hours 2 --jobs 4
```

Hết ngân sách nhưng không có `qualified.json` nghĩa là **chưa tìm được model
thắng thuyết phục**, không được gọi champion phát triển là model đã vượt cổng.
