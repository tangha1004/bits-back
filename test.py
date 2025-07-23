import rans
import numpy as np

# Định nghĩa phân phối xác suất cho 4 ký hiệu
symbols = ['A', 'B', 'C', 'D']
probs = [0.1, 0.2, 0.3, 0.4]
precision = 8  # Số bit cho xác suất

# Tính start và prob cho từng symbol
cumprob = np.cumsum([0] + probs)
print(cumprob)
starts = [int(cumprob[i] * (1 << precision)) for i in range(len(symbols))]
print(starts)
probs_int = [int(p * (1 << precision)) for p in probs]

def statfun(symbol):
    idx = symbols.index(symbol)
    return starts[idx], probs_int[idx]

def statfun_inv(cf):
    for idx, (start, prob) in enumerate(zip(starts, probs_int)):
        if start <= cf < start + prob:
            return symbols[idx], (start, prob)
    raise ValueError("cf out of range")

# Chuỗi cần mã hóa
data = ['A', 'B', 'C', 'D']

# Khởi tạo trạng thái rANS
msg = rans.msg_init

# Mã hóa từng symbol (append)
for symbol in reversed(data):  # rANS mã hóa ngược
    msg = rans.append(msg, *statfun(symbol), precision)

# Chuyển sang bitstream
bitstream = rans.flatten(msg)
print("Bitstream:", bitstream)

# Giải nén lại (pop)
msg2 = rans.unflatten(bitstream)
decoded = []
for _ in range(len(data)):
    msg2, symbol = rans.pop(msg2, statfun_inv, precision)
    decoded.append(symbol)

print("Decoded:", decoded[::-1])  # Đảo ngược lại thứ tự