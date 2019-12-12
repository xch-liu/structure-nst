require 'torch'
require 'nn'


local M = {}


--Function Set up GPU
--CPU or GPU
function M.edge_setup_gpu(gpu, backend)
  local multigpu = false
  if gpu:find(',') then
    multigpu = true
    gpu = gpu:split(',')
    for i = 1, #gpu do
      gpu[i] = tonumber(gpu[i]) + 1
    end
  else
    gpu = tonumber(gpu) + 1
  end
  local dtype = 'torch.FloatTensor'
  if multigpu or gpu > 0 then
    if backend ~= 'clnn' then
      require 'cutorch'
      require 'cunn'
      if multigpu then
        cutorch.setDevice(gpu[1])
      else
        cutorch.setDevice(gpu)
      end
      dtype = 'torch.CudaTensor'
    else
      require 'clnn'
      require 'cltorch'
      if multigpu then
        cltorch.setDevice(gpu[1])
      else
        cltorch.setDevice(gpu)
      end
      dtype = torch.Tensor():cl():type()
    end
  else
    backend = 'nn'
  end

  if backend == 'cudnn' then
    require 'cudnn'
    if cudnn_autotune then
      cudnn.benchmark = true
    end
    cudnn.SpatialConvolution.accGradParameters = nn.SpatialConvolutionMM.accGradParameters -- ie: nop
  end
  return dtype, multigpu
end

return M
