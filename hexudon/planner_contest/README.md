# Gói Chiến Thuật Thi Đấu Mới (Optimal Contest Planner)

## Luồng hiện hành và kiểm chứng cải tiến

**Chiến thuật beam mới đã bật mặc định trong `planner_contest`.** Không cần
flag riêng, model JSON hay train lại. Các lệnh `web`, `play`, `sandbox` chọn
`--planner contest` sẽ dùng luồng này; `--planner alns` vẫn dùng planner cũ.
`plan_day(..., enable_dense=False)` chỉ để tắt nhánh beam khi gọi Python trực
tiếp (ví dụ rollout chọn role), không phải mặc định khi chơi.

Model trong run training đã đóng băng **không** tự cập nhật mã mới. Muốn train
dựa trên planner mới, tạo run mới bằng `train init`; giữ nguyên run cũ để đối
chứng công bằng. Các hàm `improve_dense`, `beam_tours`, `rescue_coverage` và
`rescue_servings` cũng có thể import trực tiếp từ `hexudon.planner_contest`.

CLI dùng `strategy.MatchStrategist` → `engine.plan_day` → ALNS và `polish`;
`optimal.OptimalContestPlanner` là bộ giải riêng, không phải entry point mặc định.
Các mô tả ID ô và mốc thời gian bên dưới là chiến thuật lịch sử, không phải bảo
đảm đạt tối ưu cho mọi map.

- Chọn vai trò bằng rollout cả trận, chỉ so các lượt chạy đủ ngày. Khi hết thời
  gian, dùng phương án đã đánh giá hoặc công thức dự phòng.
- Policy hiện hành dùng **đúng một xe tiếp nhiên liệu**. Rollout so sánh vị trí
  xuất phát trong shortlist trung tâm làm xe tiếp; không dùng hai xe tiếp vì mất một xe
  thu udon. Đây là policy contest hiện hành, còn `optimal.py` chỉ là helper
  độc lập.
- Khi cắt điểm dư kho, giữ nguyên điểm hẹn nạp xăng và điểm chờ có chủ đích.
  Lượt ghé không có udon vẫn có thể cần thiết cho tiếp nhiên liệu.
- `polish.fill_idle_suffixes` thử cả đi tiếp và ghé thêm quán rồi quay lại vị trí
  đỗ. Nó dùng xăng tại **thời điểm bắt đầu di chuyển**, giữ lượt thu bước 1 và
  chạy simulator với `strict_fuel=True` trước khi nhận một kế hoạch.
- Trước ngày cuối, chuyến ghé rồi quay về chỉ được nhận nếu không đổi vị trí
  cuối ngày, không giảm xăng của bất kỳ xe nào và không tăng stay-step trên
  bất kỳ ô đường nào. Các chuyến một chiều giữ chính sách heuristic cũ;
  không khẳng định điểm cả trận của chúng luôn tăng.
- `bounds.day_bounds` là cận trên lạc quan; không khẳng định có tour đạt được.
- ALNS có hai mục tiêu tách biệt: điểm heuristic để khám phá, và kho nghiệm
  theo `(loại mới cả trận, loại hôm nay, phần udon, điểm heuristic)` để giữ bài cuối.
  Số loại lấy từ kết quả simulator, không từ danh sách spot dự định ghé.
- `coverage.rescue_coverage` thử chèn loại thiếu và thay phần đuôi một tour,
  tối đa 128 lần giải mã, bị chặn theo deadline. Chỉ nhận nếu L/D thực thu
  tăng; trước ngày cuối còn đòi giá trị cuối ngày không giảm. Giá trị này là
  ước lượng, nên không bảo đảm D cả trận tăng trong mọi trường hợp.
- Không nộp đè bài đã được nhận bằng bài có tuple L/D/Q thấp hơn chỉ vì heuristic
  cao. Giá trị tương lai vẫn là heuristic; không khẳng định Q tối ưu toàn trận.
- Trọng số mặc định hiện tại đã cập nhật `road_park=0.0387433` từ candidate
  `g00003-c000` theo yêu cầu người dùng. Hai lượt Q03 ghi nhận **20/200/819**
  và **20/200/763**, nhưng khác xe tiếp và điều kiện tìm kiếm theo thời gian;
  không thể quy +56 Q cho trọng số này. Lượt sau ngày 20/11/2026 đạt
  **20/200/801** với cùng xe tiếp và trạng thái đường như lượt 819, replay
  khớp đủ 10 ngày. Trọng số vẫn mang tính thử nghiệm, chưa chứng minh cải thiện
  ổn định đa map. Không lấy hạng 1 khi các đội khác không nộp làm bằng chứng mạnh hơn.

### Thử nghiệm lập lịch xe tiếp — chưa bật mặc định

`tanker.py` thay đổi cách sắp điểm hẹn, không học thêm trọng số. Bật nội bộ qua
`MatchStrategist(..., enable_tanker_search=True)` hoặc
`make_context(..., tanker_order_search=True)`; mặc định vẫn **False**.

- Giữ decoder/scheduler cũ làm fallback. Chỉ thử thêm khi đúng một tanker và
  fallback bỏ điểm hẹn. Không đổi số lượng xe hoặc luật nhiên liệu.
- Thử thứ tự theo hạn chót và beam tối đa 8 cuộc hẹn, width 16. Với nhiều hẹn
  hơn, beam lùi về thứ tự hạn chót; đây không phải bộ giải tối ưu chính xác.
- Mỗi phương án được dựng lại cùng patrol và kiểm bằng simulator với
  `strict_fuel=True`. Nhiều hẹn hơn hoặc heuristic cao hơn chưa đủ: phải giữ
  mọi brand fallback đã thu và tăng tuple L/D/Q thực thu mới thay fallback.
- Chỉ bảo vệ kết quả **của cùng một DaySolution**. Việc thay decoder có thể đổi
  quỹ đạo ALNS và trạng thái ngày sau; không suy ra planner sẽ không thua cả trận.
- Cache thứ tự tối đa 128 cấu hình cửa sổ trong một context ngày. Các lượt
  decode không bỏ hẹn không chạy tìm thứ tự bổ sung. Vẫn cần đo ngân sách giây
  vì số vòng bằng nhau không đồng nghĩa chi phí tính bằng nhau.
- Hook trong `planner/decoder.py` là tùy chọn; planner cũ không truyền hook nên
  giữ hành vi cũ. Thử nghiệm đóng băng **cả** decoder dùng chung, không chỉ gói
  contest, để hai phía không vô tình cùng dùng dependency mới.

Ca kiểm tra hồi quy đã tái hiện được: thứ tự cũ thu **4 loại / 4 phần**, thứ tự
mới **5 loại / 6 phần**, đúng một tanker, không thiếu xăng. Đây là ca kiểm tra
dựng riêng, không phải trận web. Tests ở `tests/test_tanker_order_search.py`.
Snapshot: `artifacts/tanker-order-{baseline,candidate}-20261120-v1`. Bản candidate
chỉ bật flag trong snapshot thử nghiệm. Báo cáo đối chứng 8 cấu hình Mock sẽ ở
`artifacts/tanker-order-paired-mock-20261120-v1.json`; chưa có kết luận tăng điểm web.

Đối chứng v1 đã hoàn tất, 40 vòng/ngày, role 0, seed 0, traffic 59:
**1 thắng / 6 hòa / 1 thua**, không bài lỗi. Q03 tăng **20/191/521 → 20/194/531**,
nhưng Q06 giảm **15/150/809 → 15/150/704**. Đây là local trên cấu hình Mock,
không phải các lượt web mới. Exit 1 là cổng L/D/Q từ chối, không phải crash.
Nhánh vẫn **tắt mặc định**.

Tái lập ba ngày đầu Q06 cho thấy ngày 1 có cùng trạng thái cuối; ngày 2 candidate
lấy 102 thay vì 100 phần nhưng đổi vị trí đỗ và hai patrol còn 55/100 xăng,
so với 133/190 của baseline. Ngày 3 thu 51 thay vì 55 phần. Đây là điểm bắt đầu
lệch trạng thái, chưa chứng minh toàn bộ mức giảm 105 Q chỉ do nhiên liệu.
Chẩn đoán: `artifacts/tanker-order-q06-prefix-diagnostic-20261120-v1.json`.
Không coi cải thiện decoder trên cùng một nghiệm là bảo đảm cải thiện cả trận.

### Thử nghiệm nhìn trước một ngày — chưa bật mặc định

`lookahead.py` thử đổi phần chờ cuối ngày để chọn vị trí đỗ patrol/tanker.
Mục tiêu là giữ kết quả hôm nay và tạo trạng thái xuất phát tốt hơn ngày sau,
không phải đổi trọng số rồi coi một lượt web cao điểm là bằng chứng cải thiện.

- Bật nội bộ bằng `MatchStrategist(..., enable_lookahead=True)` hoặc
  `plan_day(..., enable_lookahead=True)`. Mặc định vẫn **False**, chưa nối vào
  lệnh chơi web và không thay đổi run model đã đóng băng.
- Chỉ thử khi có đúng một tanker và còn ngày kế tiếp. Chỉ sửa lệnh chờ cuối
  của một agent; giữ mọi lượt thu đã có và không giảm tuple L/D/Q hôm nay.
- Tối đa 32 lần kiểm phương án đỗ; chọn tối đa 3 phương án để so với incumbent.
  Mỗi dự báo ngày sau dùng cùng seed, 8 vòng ALNS, kho được bổ sung lại, vị trí
  và xăng được chuyển tiếp đúng. Tắt beam/lookahead trong dự báo để tránh đệ quy
  và giới hạn chi phí. Đây là xấp xỉ, không phải planner đầy đủ của ngày mai.
- Dùng forecast sẵn có và cận dưới traffic từ stay-step của phương án; không
  đọc trạng thái đường thật của ngày tương lai để ra quyết định.
- Chỉ chọn nếu tuple dự báo hai ngày tăng. Hòa, dự báo sai luật hoặc không hoàn
  tất so sánh trước deadline thì giữ incumbent. Một lần dự báo không bị ngắt
  giữa chừng; vì vậy vẫn phải đo độ trễ trước khi thử online.
- Cho phép gửi phương án đã qua so sánh có cùng Q hôm nay nhưng tốt hơn theo
  dự báo, dù điểm heuristic cũ thấp hơn. Vẫn giữ giới hạn số bài gửi và chặn
  mọi bài có L/D/Q hôm nay thấp hơn bài đã được nhận.

Snapshot đối chứng của thử nghiệm đầu: `artifacts/lookahead-baseline-20261120-v1`
và `artifacts/lookahead-candidate-20261120-v1`. Cấu hình 8 map lấy từ capture
web nằm ở `artifacts/lookahead-mock-maps-20261120-v1` kèm manifest nguồn.
Lệnh đối chứng đang dùng ngân sách vòng cố định; không được coi đó là cổng
thời gian web. Chỉ xem là cải thiện sau khi có kết quả toàn trận, kiểm tra
ngân sách giây, rồi kiểm chứng Mock lặp lại. Chưa có kết luận tăng Q toàn trận.

Kết quả thử nghiệm v1 đã hoàn tất trên 8 map, seed 0, 40 vòng/ngày, role 0,
mẫu số traffic 59: **5 thắng / 1 hòa / 2 thua**, không có bài lỗi. Q05 giảm
**950 → 947 Q** và Q06 giảm **809 → 784 Q**, cùng L/D. Vì có hai cặp thua,
nhánh này **không qua cổng** và vẫn tắt mặc định, dù một số map khác tăng D/Q.
Đây là phép đo local trên cấu hình web, không phải điểm từ một lượt web mới.
Báo cáo: `artifacts/lookahead-paired-mock-20261120-v1.json`.

### Vi tối ưu cho map nhiều udon

`dense_search.improve_dense` bổ sung beam search bài toán orienteering trên
đồ thị có hướng: state gồm tập quán đã ghé, quán cuối, số bước và xăng. Những
state trùng `(visited,last)` bị loại nếu một state khác không tốn nhiều bước
và xăng hơn; frontier được giới hạn theo beam width.

- Chỉ bật khi **mọi quán đủ stock cho tất cả patrol** và ít nhất một nửa các
  brand có nhiều quán. Không dùng beam đếm quán trên map đa số brand đơn lẻ;
  thử nghiệm rộng hơn đã làm mất D ở ngày sau nên bị loại.
- Chọn nhiều điểm kết thúc khác nhau, không chỉ 16 tour ngắn nhất cùng một vùng.
- Chỉ dùng xăng thật đầu ngày để đề xuất route. Bộ giải mã toàn đội kiểm lại
  lệnh, stock và các điểm hẹn tanker được chuyển theo quán trong route mới.
- Chỉ nhận nếu L/D/Q hôm nay tăng. Trước ngày cuối bảo vệ dự trữ xăng và giới
  hạn mất giá trị cuối ngày; không bảo đảm điểm toàn trận trên mọi map mới.
- Beam width 2048 khi không dùng deadline hoặc còn >=6 giây; nếu ngắn thì 512.
  ALNS chừa tối đa 8 giây cho hậu xử lý ở cửa sổ >=30 giây, tối đa 3 giây cho
  cửa sổ ngắn hơn. Không chạy beam trong rollout chọn role để tránh lấn cửa sổ.
- Đã sửa ALNS giữ lệnh chờ `stop_idx=-1` khi xóa điểm, và lưu nghiệm tốt ngay
  ở bước restart cuối. Đã chặn nộp đè bài giảm Q dù heuristic cao.

Thử nghiệm từng nhánh trên map `2dd1ab59`, 1000 vòng/ngày, role 80 vòng, mẫu số
giao thông 59: baseline 15/150/**939**, beam 512: **949**, beam 2048: **951**,
beam 2048 đa dạng endpoint: **952**. Đây là các phép đấu offline paired, không
phải replay thi mới và không phải chứng minh 960 là cận tối ưu. `--players 59`
chỉ đổi mẫu số lưu lượng; đối thủ vẫn là các bot giả lập chứ không phải 59
lộ trình thật. Ở phép đo 5 giây/ngày, nhánh beam 512 tăng Q **935 → 943**.
Báo cáo các nhánh: `artifacts/micro-dense-{target,width-ab,endpoints-ab,5seconds}.json`.

**Bản cuối của vòng này** được đóng băng ở `artifacts/micro-20261120-final`:

| Bộ đo | Baseline | Candidate | Kết quả |
|---|---|---|---|
| `2dd1ab59`, seed 0, 1000 vòng/ngày, role 80, players 59 | 15/150/939 | **15/150/952** | +13 Q, giữ đủ loại cả 10 ngày |
| 8 map khác, seed 0, 300 vòng/ngày, role 40, players 4 | 142/984/2627 | 142/984/2627 | 8 hòa, 0 thua theo L/D/Q |
| `2dd1ab59`, seed 0, 5 giây/ngày, players 59 | 15/150/936 | **15/150/943** | +7 Q |

Không có bài lỗi. Đây là một số seed/budget đã thử, không chứng minh không thua
trên mọi map hay theo tiêu chí thời gian phản hồi khi điểm hòa. Mức **960 chưa
đạt được**, cũng chưa được chứng minh là cận trên. Báo cáo đúng bản cuối:
`artifacts/micro-final-{target,batch,5seconds}.json`.

Nhánh `local_search.py` (2-opt + relocate + fill) được giữ để thử nghiệm, **không
gọi từ engine mặc định**: nó tăng 1 Q trên map mục tiêu nhưng làm giảm D ở map
khác. Không dùng kết quả nhánh đó để tuyên bố cải thiện của bản mặc định.

Để đấu hai phiên bản đóng băng (mỗi thư mục chứa các file `.py` của gói này):

```powershell
python -m hexudon.offline.contest_compare --baseline artifacts/round2-baseline --candidate artifacts/round2-safe-excursions --iterations 300 --role-iterations 40 --out artifacts/comparison.json
```

For a strict paired-test gate, add `--fail-on-loss`; the command exits nonzero
if the candidate loses under contest ranking (L/D/Q, then response time). Run the gate both with deterministic
iterations and with the contest wall-clock budget before accepting a strategy
change. This protects only the map/seed/budget matrix actually tested.

Báo cáo có hash của **cả hai** gói. Không chỉnh thư mục snapshot trong lúc chạy.
Điểm thắng/hòa/thua theo `(L,D,Q)`; ngân sách vòng lặp không đồng nghĩa cùng thời
gian CPU. Dùng thêm `--seconds 1` để so trong ngân sách thời gian thực.

### Đo chuyến đi–về (so với bản trước vòng cải tiến này)

Hai gói được đóng băng ở `artifacts/round2-baseline` và
`artifacts/round2-safe-excursions`; báo cáo ghi hash Python của cả hai.

| Bộ đo | Thắng / hòa / thua | L/D/Q baseline | L/D/Q candidate |
|---|---|---|---|
| 5 map × 2 seed, 300 vòng/ngày | 2 / 8 / 0 | 164 / 919 / 1546 | 164 / 919 / 1548 |
| 3 map replay + random8x32 + random28, 1 seed | 0 / 5 / 0 | 83 / 775 / 2206 | 83 / 775 / 2206 |
| sample24, random16, random24, 1 giây/ngày | 0 / 2 / 1 | 46 / 239 / 456 | 46 / 239 / 454 |

Không có bài lỗi trong các lượt đo này. Lợi ích Q còn nhỏ; không suy ra mạnh
hơn dưới mọi hạn thời gian hay đạt trần mọi ngày. Bản thử đi–về không ràng buộc
xăng/traffic làm giảm điểm toàn trận và **không** được giữ trong mã hiện hành.
Báo cáo: `artifacts/round2-safe-{direct,holdout,wallclock}.json`.

### Đo ưu tiên loại và cứu coverage

So bản đóng băng `artifacts/coverage-baseline` với
`artifacts/coverage-safe-rescue` (hash Python trùng mã hiện hành lúc đo):

| Bộ đo | Thắng / hòa / thua | L/D/Q baseline | L/D/Q candidate | Ngày đủ loại |
|---|---|---|---|---|
| 9 map, seed 0, 300 vòng/ngày | 4 / 4 / 1 | 145 / 973 / 2262 | 145 / 980 / 2331 | 48 → 50 / 62 |
| random24 + 2 replay, seed 1 | 2 / 0 / 1 | 55 / 540 / 1364 | 55 / 544 / 1563 | 24 → 25 / 30 |
| 3 map, 1 giây/ngày | 2 / 0 / 1 | 47 / 361 / 961 | 47 / 363 / 917 | 19 → 19 / 24 |

Không có bài lỗi. Các cặp thua vẫn có thể bằng L/D nhưng ít Q hơn. Đây là mẫu
nhỏ, không phải chứng minh tối ưu hoặc tăng trên mọi map. Ngày đủ loại chỉ đếm
đủ tất cả brand của map, không suy ra mọi ngày còn thiếu đều có nghiệm phủ đủ.
Bản cứu loại không kiểm tra giá trị cuối ngày đã bị loại vì làm giảm tổng D.
Báo cáo: `artifacts/coverage-safe-{comparison,holdout,wallclock}.json`.

---

Thư mục này (`hexudon/planner_contest/`) chứa toàn bộ các cải tiến và thuật toán tối ưu hóa mới nhằm đánh bại đối thủ (Team 77 - DAU 1), được tách biệt hoàn toàn khỏi thư mục chiến thuật cũ (`hexudon/planner/`).

---

## 1. So sánh: Chiến thuật Cũ vs Chiến thuật Mới

| Đặc tính | Chiến thuật Cũ (`hexudon/planner/`) | Chiến thuật Mới (`hexudon/planner_contest/`) |
| :--- | :--- | :--- |
| **Bán kính ứng viên (`NEAR_K`)** | `NEAR_K = 10` (bị mù trước các quán xa/góc như ô 808) | `NEAR_K = 30` + Luôn ép các quán của nhãn hàng chưa phủ vào ứng viên |
| **Tiền xử lý đồ thị** | Chạy Dijkstra 900 ô lặp đi lặp lại trong quá trình tìm kiếm | `precompute_all_pairs`: Dijkstra 35 nút quan trọng trong $O(1)$, chạy $< 0.08$s |
| **Phân tầng theo độ dài ngày** | Không phân biệt ngày ngắn hay dài, dễ sa vào bẫy vùng thưa | Phân tầng rõ rệt: $T \le 90$ (ngắn), $T \le 155$ (vừa - bỏ vùng thưa 142/197/233), $T \ge 170$ (dài - Full-map TSP) |
| **Điều phối xe bồn tiếp xăng** | Thụ động đứng một chỗ (ô 321 hoặc theo điểm neo đơn lẻ) | Zero-Wait Rendezvous: Cắm chốt tại nút cổ chai 475 ở ngày dài, hoặc 579/487/831 ở ngày ngắn |
| **Tận dụng bước thừa cuối ngày** | Xe có thể đứng im 10–19 bước cuối ngày do cắt tỉa sớm | `Greedy Insertion` vét cạn: quét toàn bộ ô chưa ăn chèn vào bước/xăng thừa |
| **Đỗ xe cuối ngày** | Dễ dồn 3–4 xe vào ngõ cụt 34 làm tê liệt ngày sau | Anti-Clumping ngõ cụt 34: cấm đỗ quá 1 xe ở ngõ cụt, ép đỗ vào Cụm Core |

---

## 2. Cách Chuyển Đổi Giữa 2 Chiến Thuật

### Cách 1: Chạy qua dòng lệnh (CLI Flag)
Bạn có thể dùng cờ `--planner` khi chạy `hexudon`:
```powershell
# Chạy chiến thuật MỚI (Mặc định):
python -m hexudon play ... --planner contest

# Chạy lại chiến thuật CŨ:
python -m hexudon play ... --planner alns
```

### Cách 2: Thiết lập qua Biến Môi Trường (Environment Variable)
```powershell
# Chọn chiến thuật MỚI:
$env:HEXUDON_PLANNER="contest"

# Quay lại chiến thuật CŨ:
$env:HEXUDON_PLANNER="alns"
```

### Cách 3: Nhập khẩu trực tiếp trong Python Code
```python
# Sử dụng chiến thuật CŨ (ALNS):
from hexudon.planner.strategy import MatchStrategist

# Sử dụng chiến thuật MỚI (Optimal Contest):
from hexudon.planner_contest.strategy import MatchStrategist
# hoặc dùng trực tiếp planner:
from hexudon.planner_contest.optimal import OptimalContestPlanner
```
