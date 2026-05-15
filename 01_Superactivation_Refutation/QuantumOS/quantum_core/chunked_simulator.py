import torch
import math

_GLOBAL_CHUNKS = None

class PyTorchChunkedSimulator:
    def __init__(self, num_qubits, chunk_bits=24, device='cuda'):
        global _GLOBAL_CHUNKS
        self.num_qubits = num_qubits
        self.chunk_bits = min(chunk_bits, num_qubits)
        self.num_chunks = 1 << (num_qubits - self.chunk_bits)
        self.device = device
        
        # Використовуємо глобальний буфер, щоб не виділяти 8 ГБ пам'яті для кожного прогону (уникнення RAM Leak)
        if _GLOBAL_CHUNKS is None or len(_GLOBAL_CHUNKS) != self.num_chunks:
            _GLOBAL_CHUNKS = [
                torch.zeros(1 << self.chunk_bits, dtype=torch.complex64, device='cpu').pin_memory()
                for _ in range(self.num_chunks)
            ]
            
        self.chunks = _GLOBAL_CHUNKS
        
        # Швидке очищення вектора до стану |0...0>
        for i in range(self.num_chunks):
            self.chunks[i].zero_()
        self.chunks[0][0] = 1.0 + 0.0j

    def apply_1q_gate(self, matrix, target):
        matrix = matrix.to(self.device).to(torch.complex64)
        if target < self.chunk_bits:
            for i in range(self.num_chunks):
                chunk = self.chunks[i].to(self.device, non_blocking=True)
                dim1 = 1 << (self.chunk_bits - target - 1)
                dim3 = 1 << target
                chunk = chunk.view(dim1, 2, dim3)
                
                out = torch.empty_like(chunk)
                out[:, 0, :] = matrix[0, 0] * chunk[:, 0, :] + matrix[0, 1] * chunk[:, 1, :]
                out[:, 1, :] = matrix[1, 0] * chunk[:, 0, :] + matrix[1, 1] * chunk[:, 1, :]
                
                self.chunks[i].copy_(out.view(-1), non_blocking=True)
        else:
            step = 1 << (target - self.chunk_bits)
            for i in range(self.num_chunks):
                if (i & step) == 0:
                    idx0 = i
                    idx1 = i | step
                    chunk0 = self.chunks[idx0].to(self.device, non_blocking=True)
                    chunk1 = self.chunks[idx1].to(self.device, non_blocking=True)
                    
                    out0 = matrix[0, 0] * chunk0 + matrix[0, 1] * chunk1
                    out1 = matrix[1, 0] * chunk0 + matrix[1, 1] * chunk1
                    
                    self.chunks[idx0].copy_(out0, non_blocking=True)
                    self.chunks[idx1].copy_(out1, non_blocking=True)
            torch.cuda.synchronize()

    def apply_cnot(self, control, target):
        if control < self.chunk_bits and target < self.chunk_bits:
            for i in range(self.num_chunks):
                chunk = self.chunks[i].to(self.device, non_blocking=True)
                q_max = max(control, target)
                q_min = min(control, target)
                
                dim1 = 1 << (self.chunk_bits - q_max - 1)
                dim3 = 1 << (q_max - q_min - 1)
                dim5 = 1 << q_min
                
                chunk = chunk.view(dim1, 2, dim3, 2, dim5)
                
                if control > target:
                    temp = chunk[:, 1, :, 0, :].clone()
                    chunk[:, 1, :, 0, :] = chunk[:, 1, :, 1, :]
                    chunk[:, 1, :, 1, :] = temp
                else:
                    temp = chunk[:, 0, :, 1, :].clone()
                    chunk[:, 0, :, 1, :] = chunk[:, 1, :, 1, :]
                    chunk[:, 1, :, 1, :] = temp
                
                self.chunks[i].copy_(chunk.view(-1), non_blocking=True)
        elif control >= self.chunk_bits and target < self.chunk_bits:
            step_c = 1 << (control - self.chunk_bits)
            for i in range(self.num_chunks):
                if (i & step_c) != 0:
                    chunk = self.chunks[i].to(self.device, non_blocking=True)
                    dim1 = 1 << (self.chunk_bits - target - 1)
                    dim3 = 1 << target
                    chunk = chunk.view(dim1, 2, dim3)
                    
                    temp = chunk[:, 0, :].clone()
                    chunk[:, 0, :] = chunk[:, 1, :]
                    chunk[:, 1, :] = temp
                    
                    self.chunks[i].copy_(chunk.view(-1), non_blocking=True)
        elif control < self.chunk_bits and target >= self.chunk_bits:
            step_t = 1 << (target - self.chunk_bits)
            for i in range(self.num_chunks):
                if (i & step_t) == 0:
                    idx0 = i
                    idx1 = i | step_t
                    chunk0 = self.chunks[idx0].to(self.device, non_blocking=True)
                    chunk1 = self.chunks[idx1].to(self.device, non_blocking=True)
                    
                    dim1 = 1 << (self.chunk_bits - control - 1)
                    dim3 = 1 << control
                    
                    chunk0 = chunk0.view(dim1, 2, dim3)
                    chunk1 = chunk1.view(dim1, 2, dim3)
                    
                    temp = chunk0[:, 1, :].clone()
                    chunk0[:, 1, :] = chunk1[:, 1, :]
                    chunk1[:, 1, :] = temp
                    
                    self.chunks[idx0].copy_(chunk0.view(-1), non_blocking=True)
                    self.chunks[idx1].copy_(chunk1.view(-1), non_blocking=True)
        else:
            step_c = 1 << (control - self.chunk_bits)
            step_t = 1 << (target - self.chunk_bits)
            for i in range(self.num_chunks):
                if (i & step_c) != 0 and (i & step_t) == 0:
                    idx0 = i
                    idx1 = i | step_t
                    temp = self.chunks[idx0].clone()
                    self.chunks[idx0].copy_(self.chunks[idx1])
                    self.chunks[idx1].copy_(temp)

    def apply_ry(self, angle, target):
        cos_a = torch.cos(angle / 2)
        sin_a = torch.sin(angle / 2)
        matrix = torch.stack([
            torch.stack([cos_a, -sin_a]),
            torch.stack([sin_a, cos_a])
        ])
        self.apply_1q_gate(matrix, target)

    def apply_rx(self, angle, target):
        cos_a = torch.cos(angle / 2).to(torch.complex64)
        sin_a = torch.sin(angle / 2).to(torch.complex64)
        zero = torch.tensor(0.0, dtype=torch.complex64, device=angle.device)
        minus_i = torch.tensor(-1j, dtype=torch.complex64, device=angle.device)
        matrix = torch.stack([
            torch.stack([cos_a, minus_i * sin_a]),
            torch.stack([minus_i * sin_a, cos_a])
        ])
        self.apply_1q_gate(matrix, target)

    def apply_rz(self, angle, target):
        exp_minus = torch.exp(-1j * angle / 2)
        exp_plus = torch.exp(1j * angle / 2)
        zero = torch.tensor(0.0, dtype=torch.complex64, device=angle.device)
        matrix = torch.stack([
            torch.stack([exp_minus, zero]),
            torch.stack([zero, exp_plus])
        ])
        self.apply_1q_gate(matrix, target)

    def expval_z(self):
        expvals = []
        for target in range(self.num_qubits):
            expval = 0.0
            if target < self.chunk_bits:
                for i in range(self.num_chunks):
                    chunk = self.chunks[i].to(self.device, non_blocking=True)
                    dim1 = 1 << (self.chunk_bits - target - 1)
                    dim3 = 1 << target
                    chunk = chunk.view(dim1, 2, dim3)
                    
                    prob0 = torch.sum(torch.abs(chunk[:, 0, :])**2)
                    prob1 = torch.sum(torch.abs(chunk[:, 1, :])**2)
                    expval += (prob0 - prob1).item()
            else:
                step_t = 1 << (target - self.chunk_bits)
                for i in range(self.num_chunks):
                    chunk = self.chunks[i].to(self.device, non_blocking=True)
                    prob = torch.sum(torch.abs(chunk)**2).item()
                    if (i & step_t) == 0:
                        expval += prob
                    else:
                        expval -= prob
            expvals.append(expval)
        return torch.tensor(expvals, dtype=torch.float64, device=self.device)
