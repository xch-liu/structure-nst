require 'torch'
require 'nn'
require 'image'
require 'optim'

require 'loadcaffe'


local cmd = torch.CmdLine()

--Basic options
cmd:option('-proto_file', 'hed_deploy.prototxt')
cmd:option('-model_file', 'hed_pretrained_bsds.caffemodel')

--cmd:option('-backend', 'cuda', 'cuda|opencl')
cmd:option('-backend', 'nn', 'nn|cudnn|clnn')
cmd:option('-edge_layers', 'relu5_3', 'layers for edge')
cmd:option('-content_image', 'out.png', 'Content target image')
cmd:option('-synthesised_image', 'out_900.png', 'Synthesised target image')
cmd:option('-image_size_width', 500, 'Maximum height / width of generated image')
cmd:option('-image_size_height', 500, 'Maximum height / width of generated image')
cmd:option('-edge_weight', 1e2)
cmd:option('-init', 'random', 'random|image')
cmd:option('-init_image', '')
cmd:option('-gpu', '0', 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')
cmd:option('-multigpu_strategy', '', 'Index of layers to split the network across GPUs')
cmd:option('-seed', -1)
cmd:option('-normalize_gradients', false)


local function main(params)
    local dtype, multigpu = setup_gpu(params)

    local loadcaffe_backend = params.backend
    if params.backend == 'clnn' then loadcaffe_backend = 'nn' end

    local cnn = loadcaffe.load(params.proto_file, params.model_file, loadcaffe_backend):type(dtype)
    
    --Preprocess content image
    local content_image = image.load(params.content_image, 3)
    content_image = image.scale(content_image, params.image_size_width, params.image_size_height, 'bilinear')
    local content_image_caffe = preprocess(content_image):float()

    --Preprocess synthesised image
    local synthesised_image = image.load(params.synthesised_image, 3)
    synthesised_image = image.scale(synthesised_image, params.image_size_width, params.image_size_height, 'bilinear')
    local synthesised_image_caffe = preprocess(synthesised_image):float()

    --Set up the network
    local edge_layers = params.edge_layers:split(",")
   
    print (edge_layers) 
    print (#edge_layers) 

   
    local next_edge_idx = 1
    local net = nn.Sequential()
    
    for i = 1, #cnn do
      if next_edge_idx <= #edge_layers then
        local layer = cnn:get(i)
        local name = layer.name      
        net:add(layer)     
        if name == edge_layers[next_edge_idx] then
          net:add(layer)
          next_edge_idx = next_edge_idx + 1
        end
      end
    end
    
    print 'The net has been built'
    print (net)
 
    --Capture content image
    print 'Capturing content targets'
    content_image_caffe = content_image_caffe:type(dtype)
    local out_edge1 = net:forward(content_image_caffe:type(dtype))
    local reshape_edge1 = torch.reshape(out_edge1, 512, 36*36)

    --Capture synthesised image
    print 'Capturing synthesised targets'
    synthesised_image_caffe = synthesised_image_caffe:type(dtype)
    local out_edge2 = net:forward(synthesised_image_caffe:type(dtype))
    print 'The size of out_edge2 is:'
    print (out_edge2:size())
    local reshape_edge2 = torch.reshape(out_edge2, 512, 36*36)
    print (reshape_edge2:size())

    --Compute the edge loss
    local edge_loss = torch.dist(reshape_edge1, reshape_edge2)
    --local edge_loss = torch.dist(out_edge1, out_edge2)
    print (edge_loss)
    
    print('Hello! HaHaHaHa...')


end


-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel.
function preprocess(img)
  --local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(256.0)
  --mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  --img:add(-1, mean_pixel)
  return img
end


--Function Set up GPU
--CPU or GPU
function setup_gpu(params)
  local multigpu = false
  if params.gpu:find(',') then
    multigpu = true
    params.gpu = params.gpu:split(',')
    for i = 1, #params.gpu do
      params.gpu[i] = tonumber(params.gpu[i]) + 1
    end
  else
    params.gpu = tonumber(params.gpu) + 1
  end
  local dtype = 'torch.FloatTensor'
  if multigpu or params.gpu > 0 then
    if params.backend ~= 'clnn' then
      require 'cutorch'
      require 'cunn'
      if multigpu then
        cutorch.setDevice(params.gpu[1])
      else
        cutorch.setDevice(params.gpu)
      end
      dtype = 'torch.CudaTensor'
    else
      require 'clnn'
      require 'cltorch'
      if multigpu then
        cltorch.setDevice(params.gpu[1])
      else
        cltorch.setDevice(params.gpu)
      end
      dtype = torch.Tensor():cl():type()
    end
  else
    params.backend = 'nn'
  end

  if params.backend == 'cudnn' then
    require 'cudnn'
    if params.cudnn_autotune then
      cudnn.benchmark = true
    end
    cudnn.SpatialConvolution.accGradParameters = nn.SpatialConvolutionMM.accGradParameters -- ie: nop
  end
  return dtype, multigpu
end


local params = cmd:parse(arg)
main(params)
