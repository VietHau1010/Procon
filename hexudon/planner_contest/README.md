# Gói Chiến Thuật Thi Đấu Mới (Optimal Contest Planner)

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
