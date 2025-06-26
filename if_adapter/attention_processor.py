# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x):
        max = torch.max(x,1)[0].unsqueeze(1)
        avg = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max,avg), dim=1)
        output = self.conv(concat)
        output = F.sigmoid(output) * x 
        return output 

class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels//self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.r, out_features=self.channels, bias=True))

    def forward(self, x):
        max = F.adaptive_max_pool2d(x, output_size=1)
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        linear_max = self.linear(max.view(b,c)).view(b, c, 1, 1)
        linear_avg = self.linear(avg.view(b,c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = F.sigmoid(output) * x
        return output
    
class CBAM(nn.Module):
    def __init__(self, channels, r):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)

    def forward(self, x):
        output = self.cam(x)
        output = self.sam(output)
        return output



class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)



class AttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        phrase_num_arr=None,
        ap_tokens=None,
        boxes=None,
        use_cond=True,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class IFAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, num_heads=None, scale=1.0, num_tokens=4, enable_alpha=False, use_gate=False, max_obj=8):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens
        self.use_gate = use_gate
        self.max_obj = max_obj


        self.to_k_ap = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ap = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

        self.q_norm = GemmaRMSNorm(hidden_size // num_heads)
        self.k_norm = GemmaRMSNorm(hidden_size// num_heads)

        self.cbam = CBAM(hidden_size, 16)
        self.conv = nn.Conv2d(hidden_size, 1, 1, 1)


        self.enable_alpha = enable_alpha
        if enable_alpha:
            self.alpha = nn.Parameter(torch.tensor(0.))
           
    def map_construction(self, bboxes_list, area, batch_size, num_heads, head_dim, dtype, device):
        """
        input:
        output: (B, heads, L, n_tokens, dim)
        """
        num_block_row_col = int(area ** 0.5)
        all_sample_background_mask_list = []
        all_sample_cond_mask_list = [] # for all samples
        mask_scale = []
        
        for bboxes in bboxes_list:
            single_sample_cond_mask = []
            single_sample_mask_scale = []

            for i in range(bboxes.shape[0]):
                single_box_cond_mask = torch.zeros((1, num_block_row_col, num_block_row_col), dtype=dtype, device=device)

                current_box_start = torch.floor(bboxes[i, 0:2] * num_block_row_col)
                current_box_end = torch.ceil(bboxes[i, 2:4] * num_block_row_col)

                top_left_x = int(current_box_start[0])
                top_left_y = int(current_box_start[1])
                bottom_right_x = int(current_box_end[0])
                bottom_right_y = int(current_box_end[1])

                single_box_cond_mask[:, top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 1
                single_sample_cond_mask.append(single_box_cond_mask)

            single_sample_cond_mask_torch = torch.concat(single_sample_cond_mask)
            background_map = (single_sample_cond_mask_torch.sum(axis=0)< 1).int()

            box_areas = single_sample_cond_mask_torch.sum(axis=[1,2])
            # box_area_scale = box_areas.max()/box_areas # 把box
            box_area_scale = (area - background_map.sum())/box_areas # 把box

            mask_scale.append(box_area_scale)
            all_sample_background_mask_list.append(background_map[None, :, :])
            all_sample_cond_mask_list.append(single_sample_cond_mask_torch)

        # mask_scale = torch.cat(mask_scale)
        # all_sample_cond_mask_list = torch.cat(all_sample_cond_mask_list)
        # all_sample_background_mask_list = torch.cat(all_sample_background_mask_list)

        return all_sample_cond_mask_list, all_sample_background_mask_list, mask_scale
        

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        phrase_num_arr=None,
        ap_tokens=None,
        boxes=None,
        use_cond=True,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        # ------------------------------------ to start with ---------------------------------------- #


        ap_key = self.to_k_ap(ap_tokens)
        ap_value = self.to_v_ap(ap_tokens)
        
        # pad to fix length
        # ap_key_list = list(torch.split(ap_key, phrase_num_arr, dim=0))
        # ap_value_list = list(torch.split(ap_value, phrase_num_arr, dim=0))
        bboxes_list = list(torch.split(boxes, phrase_num_arr, dim=0))
        ap_key = ap_key.view(ap_key.shape[0], -1, attn.heads, head_dim).transpose(1, 2)
        ap_key = self.k_norm(ap_key)

        ap_value = ap_value.view(ap_value.shape[0], -1, attn.heads, head_dim).transpose(1, 2)
        
        all_sample_cond_mask_list, all_sample_background_mask_list, mask_scale = self.map_construction(bboxes_list, hidden_states.shape[1], hidden_states.shape[0], attn.heads, head_dim, ap_key.dtype, hidden_states.device)
        
        # split and padding
        
        ap_key_list = list(torch.split(ap_key, phrase_num_arr, dim=0))
        ap_value_list = list(torch.split(ap_value, phrase_num_arr, dim=0))
        
        ap_key_padded = []
        ap_value_padded = []

        num_tokens = ap_key.shape[2]
        padded_key_list = []
        padded_value_list = []
        cond_mask_list = []
        mask_scale_list = []


        for ap_key_one_sample, ap_value_one_sample, cond_mask_one_sample, m_scale in zip(ap_key_list, ap_value_list, all_sample_cond_mask_list, mask_scale):
            
            padded_key = torch.zeros(1, self.max_obj, attn.heads, num_tokens, head_dim).to(ap_key_one_sample) # batch maxobj, num_heads, num_tokens, hidden_dims
            padded_value = torch.zeros(1, self.max_obj, attn.heads, num_tokens, head_dim).to(ap_value_one_sample) # batch maxobj, num_heads, num_tokens, hidden_dims
            padded_mask = torch.zeros(1, self.max_obj, cond_mask_one_sample.shape[1], cond_mask_one_sample.shape[2]).to(cond_mask_one_sample) # batch maxobj, num_heads, num_tokens, hidden_dims
            padded_mask_scale = torch.zeros(1, self.max_obj).to(m_scale)

            padded_key[:, :ap_key_one_sample.shape[0], :, :, :] = padded_key[:, :ap_key_one_sample.shape[0], :, :, :] + ap_key_one_sample
            padded_value[:, :ap_value_one_sample.shape[0], :, :, :] = padded_value[:, :ap_value_one_sample.shape[0], :, :, :] + ap_value_one_sample
            padded_mask[:, :cond_mask_one_sample.shape[0], :, :] = padded_mask[:, :cond_mask_one_sample.shape[0], :, :] + cond_mask_one_sample
            padded_mask_scale[:, :m_scale.shape[0]] = padded_mask_scale[:, :m_scale.shape[0]] + m_scale

            # padded_key[:, :ap_key_one_sample.shape[0], :, :, :] = ap_key_one_sample
            # padded_value[:, :ap_value_one_sample.shape[0], :, :, :] = ap_value_one_sample
            # padded_mask[:, :cond_mask_one_sample.shape[0], :, :] = cond_mask_one_sample
            # padded_mask_scale[:, :m_scale.shape[0]] = m_scale

            padded_key_list.append(padded_key)
            padded_value_list.append(padded_value)
            cond_mask_list.append(padded_mask)
            mask_scale_list.append(padded_mask_scale)

        padded_key = torch.concat(padded_key_list)
        padded_value = torch.concat(padded_value_list)
        cond_mask = torch.concat(cond_mask_list)
        mask_scale = torch.concat(mask_scale_list)

        query = self.q_norm(query)
        # query = query.unsqueeze(1)
        query = query.unsqueeze(1).repeat(1, self.max_obj, 1,1,1)

        cond_mask = cond_mask.view(batch_size, self.max_obj, -1,1)
        attn_mask = cond_mask.view(batch_size, self.max_obj, 1, -1,1).repeat(1,1, attn.heads,1,num_tokens).to(torch.bool)
        
        attn_mask = attn_mask.to(dtype=ap_key.dtype)
        attn_mask = (1.0 - attn_mask) * torch.finfo(ap_key.dtype).min # False -》 -inf

        ap_hidden_states = F.scaled_dot_product_attention(
            query, padded_key, padded_value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )

        
        ap_hidden_states = ap_hidden_states.transpose(2,3).reshape(batch_size, self.max_obj, -1, attn.heads * head_dim)
        ap_hidden_states = ap_hidden_states * cond_mask

        cond_scale = ap_hidden_states.view(batch_size * self.max_obj, -1, attn.heads * head_dim).transpose(1,2)
        num_blocks = int(hidden_states.shape[1] ** 0.5)
        cond_scale = cond_scale.view(-1, attn.heads * head_dim, num_blocks,num_blocks)
        cond_scale = self.cbam(cond_scale)
        cond_scale = self.conv(cond_scale) # B*maxobj,num_blocks,num_blocks

        mask_scale = mask_scale.view(-1,1,1,1)
        cond_scale = (cond_scale * F.sigmoid(mask_scale)).view(batch_size, self.max_obj, hidden_states.shape[1]).softmax(dim=1)

        ap_hidden_states = (ap_hidden_states*cond_scale.unsqueeze(-1)).sum(dim=1)

        all_sample_background_mask = (1-torch.concat(all_sample_background_mask_list)).view(batch_size, -1, 1)

        ap_hidden_states = ap_hidden_states * all_sample_background_mask
        ap_hidden_states = ap_hidden_states.to(query.dtype)

        if use_cond:
            if self.enable_alpha:
                hidden_states = hidden_states + torch.tanh(self.alpha)*self.scale * ap_hidden_states
            else:
                hidden_states = hidden_states + self.scale * ap_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
