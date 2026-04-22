# 改进版的卷积长短期记忆网络，引入差分机制来提高稀疏性
import torch.nn as nn
import torch
import os

# 创建日志保存稀疏率信息
log_dir = 'log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# ConvLSTM的基本单元
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        创建一个卷积层，将输入和隐藏状态拼接后输出为4倍隐藏维度（分别对应i、f、o、g门）。
        
        Parameters：
        input_dim: int，输入张量的通道数
        hidden_dim: int，隐藏状态的通道数
        kernel_size: (int, int)，卷积核大小
        bias: bool，是否使用偏置项
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2     # 计算填充大小以保持特征图尺寸不变
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,     # 对应输入门i、遗忘门f、输出门o、候选状态g
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    # 前向传播，引入差分机制
    def forward(self, input_tensor, input_tensor_pre, cur_state):
        h_cur, c_cur, h_pre  = cur_state
        delta = h_cur - h_pre   # 计算当前隐藏状态与前一时刻隐藏状态的差值，状态差分
        delta_inp =input_tensor - input_tensor_pre  # 计算当前输入与前一时刻输入的差值，输入差分
        threshold = torch.tensor(0.002).cuda()
        # 低于阈值的差分被置零，这种机制可过滤微小变化，减少噪声影响并提高稀疏性
        delta = torch.where(delta < threshold, torch.tensor(0.0).cuda(), delta)
        delta_inp = torch.where(delta_inp < threshold, torch.tensor(0.0).cuda(), delta_inp)
        
        combined = torch.cat([delta_inp, delta], dim=1)  # 特征拼接，将输入差分和状态差分沿通道轴拼接
        if self.training == False:
            # 计算并记录稀疏率，仅在非训练模式下
            non_zero_count = torch.count_nonzero(combined).float()
            sparse_rate = (combined.numel() - non_zero_count) / combined.numel()    # 计算整体稀疏率
            sparse_rate_inp = (delta_inp.numel() - torch.count_nonzero(delta_inp).float()) / delta_inp.numel()  # 计算输入差分的稀疏率
            sparse_rate_delta = (delta.numel() - torch.count_nonzero(delta).float()) / delta.numel()    # 计算状态差分的稀疏率
            file_path = os.path.join(log_dir, f"xxsparse_rate_th_{threshold:.5f}.txt")
            with open(file_path, 'a') as f:     # 以追加模式打开文件，写入各项稀疏率统计信息
                f.write(f"sparse_rate: tot {sparse_rate} inp {sparse_rate_inp} delta{sparse_rate_delta} size {input_tensor.size(-1)}\n")
        # 标准LSTM门控机制
        combined_conv = self.conv(combined)     # 通过卷积层处理拼接后的差分特征
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        # 更新记忆细胞和隐藏状态
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next, h_cur, input_tensor      # 返回新状态、当前状态和当前输入

    def init_hidden(self, batch_size, image_size):
        """
        初始化隐藏状态，返回三个张量：
            ·当前隐藏状态
            ·当前细胞状态
            ·前一时刻隐藏状态
        """
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

# 多层ConvLSTM堆叠的完整网络，处理整个时间序列
class ConvLSTM(nn.Module):
    """
    Parameters:
        input_dim
        hidden_dim
        kernel_size
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: 是否使用偏置项
        return_all_layers: 返回所有层输出，否则只返回最后一层输出
        Note: Will do same padding.
        
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))                   输入：(batch_size, time_steps, channels, height, width)
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)     创建一个ConvLSTM网络，输入通道数64，隐藏通道数16，卷积核大小3*3，层数1，使用偏置项，返回所有层输出
        >> _, last_states = convlstm(x)                             前向传播，输出：(batch_size, time_steps, channels, height, width)
        >> h = last_states[0][0]                                    第0层的隐藏状态h
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)    # 检查卷积核大小的一致性

        # 将kernel_size和hidden_dim扩展为多层形式
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            # 创建多个ConvLSTMCell实例并添加到cell_list中
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters:
            input_tensor: 5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
            hidden_state: None. todo implement stateful
            Returns
        -------
        last_state_list:最终列表状态，包含每层最后时刻的(h,c)状态; layer_output:各层的输出
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)，统一使用batch_first格式：更符合PyTorch多数层的处理习惯；便于批量处理；与其他PyTorch模块兼容性更好
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()     # 时间步数和通道数在当前上下文中不需要单独使用

        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            # 层循环
            h, c, h_prev = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                if t==0:    # 对第一个时间步使用零张量作为前一时刻的输入
                    input_tensor_pre = torch.zeros_like(cur_layer_input[:, t, :, :, :]).cuda(device=input_tensor.device)
                else:
                    input_tensor_pre=cur_layer_input[:, t-1, :, :, :]
                # 通过cell_list中的相应层处理当前输入
                h, c, h_prev, input_tensor_pre = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], input_tensor_pre=input_tensor_pre,
                                                 cur_state=[h, c, h_prev])
                output_inner.append(h)  

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        """验证传入的kernel_size参数格式是否正确"""
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
