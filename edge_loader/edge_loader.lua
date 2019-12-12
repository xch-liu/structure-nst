require 'torch'
require 'nn'
require 'image'
require 'optim'

require 'loadcaffe'


local cmd = torch.CmdLine()

--Basic options
cmd:option('-proto_file', 'hed_deploy.prototxt')
cmd:option('-model_file', 'hed_pretrained_bsds.caffemodel')
cmd:option('-backend', 'nn', 'nn|cudnn|clnn')
cmd:option('-edge_layers', 'sigmoid-fuse', 'layers for edge')
cmd:option('-content_image', 'tubingen.jpg', 'Content target image')
cmd:option('-synthesised_image', 'starry_night.jpg', 'Synthesised target image')
cmd:option('-image_size', 512, 'Maximum height / width of generated image')
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
    --cnn:remove() 
    
    local synthesised_image = image.load(params.synthesised_image, 3)
    local out1 = cnn:forward(synthesised_image)
    print(out1:size())

    synthesised_image = image.scale(synthesised_image, params.image_size, 'bilinear')
    local synthesised_image_caffe = preprocess(synthesised_image):float()

    synthesised_image_caffe = synthesised_image_caffe:type(dtype)
    --net:forward(synthesised_image_caffe:type(dtype))
    local out1 = cnn:forward(synthesised_image_caffe)
    print(out1:size())

    local content_image = image.load(params.content_image, 3)
    content_image = image.scale(content_image, params.image_size, 'bilinear')
    local content_image_caffe = preprocess(content_image):float()

    local edge_layers = params.edge_layers:split(",")
    --Set up the network, inserting edge loss module
    local edge_losses = {}
    local next_edge_idx = 1
    local net = nn.Sequential()


    for i = 1, #cnn do
      if next_edge_idx <= #edge_layers then
        local layer = cnn:get(i)
        local name = layer.name
        local layer_type = torch.type(layer)
        
        local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')
        if is_pooling and params.pooling == 'avg' then
          assert(layer.padW == 0 and layer.padH == 0)
          local kW, kH = layer.kW, layer.kH
          local dW, dH = layer.dW, layer.dH
          local avg_pool_layer = nn.SpatialAveragePooling(kW, kH, dW, dH):type(dtype)
          local msg = 'Replacing max pooling at layer %d with average pooling'
          print(string.format(msg, i))
          net:add(avg_pool_layer)
        else
          net:add(layer)
        end
        
        if name == edge_layers[next_edge_idx] then
          print("Setting up content layer", i, ":", layer.name)
          local norm = params.normalize_gradients
          local loss_module = nn.EdgeLoss(params.edge_weight, norm):type(dtype)
          net:add(loss_module)
          table.insert(edge_losses, loss_module)
          next_edge_idx = next_edge_idx + 1
        end


      end
    end

    --local edge_losses = {}
    --Capture edge targets
    for i = 1, #edge_losses do
      edge_losses[i].mode = 'capture'
    end
    print 'Capturing content targets'
    print(net)
    print 'net has been printed'
    content_image_caffe = content_image_caffe:type(dtype)

    print 'forward the net'
    net:forward(content_image_caffe:type(dtype))
    print 'net has been forwarded'


    --Set all loss modules to loss mode
    for i = 1, #edge_losses do
      edge_losses[i].mode = 'loss'
      print(edge_losses)
    end

    -- We don't need the base CNN anymore, so clean it up to save memory.
    cnn = nil
    for i=1, #net.modules do
    local module = net.modules[i]
      if torch.type(module) == 'nn.SpatialConvolutionMM' then
        -- remove these, not used, but uses gpu memory
        module.gradWeight = nil
        module.gradBias = nil
      end
    end
    collectgarbage()

    --Capture synthesised image
    local synthesised_image = image.load(params.synthesised_image, 3)
    synthesised_image = image.scale(synthesised_image, params.image_size, 'bilinear')
    local synthesised_image_caffe = preprocess(synthesised_image):float()

    -- Run it through the network once to get the proper size for the gradient
    -- All the gradients will come from the extra loss modules, so we just pass
    -- zeros into the top of the net on the backward pass.
    for i = 1, #edge_losses do
      edge_losses[i].mode = 'capture'
    end
    print 'Capturing synthesised targets'
    print(net)
    print 'the net has been printed'
    synthesised_image_caffe = synthesised_image_caffe:type(dtype)
    net:forward(synthesised_image_caffe:type(dtype))

    local out1 = cnn:forward(synthesised_image_caffe:type(dtype))
    print(out1:size())

    --Set all loss modules to loss mode
    for i = 1, #edge_losses do
      edge_losses[i].mode = 'loss'
      print(edge_losses)
    end

    -- We don't need the base CNN anymore, so clean it up to save memory.
    cnn = nil
    for i=1, #net.modules do
    local module = net.modules[i]
      if torch.type(module) == 'nn.SpatialConvolutionMM' then
        -- remove these, not used, but uses gpu memory
        module.gradWeight = nil
        module.gradBias = nil
      end
    end
    collectgarbage()
end


-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel.
function preprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(256.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
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


-- Define an nn Module to compute edge loss in-place
local EdgeLoss, parent = torch.class('nn.EdgeLoss', 'nn.Module')

function EdgeLoss:__init(strength, normalize)
  parent.__init(self)
  self.strength = strength
  self.target = torch.Tensor()
  self.normalize = normalize or false
  self.loss = 0
  self.crit = nn.MSECriterion()
  self.mode = 'none'
end

function EdgeLoss:updateOutput(input)

  if self.mode == 'loss' then
    print 'Enter EdgeLoss updataOutput, mode is loss'
    self.loss = self.crit:forward(input, self.target) * self.strength
  elseif self.mode == 'capture' then
    print 'Enter EdgeLoss updataOutput, mode is capture'
    self.target:resizeAs(input):copy(input)
  end
  print 'Get out of the EdgeLoss updateOutput'
  self.output = input
  return self.output
end

function EdgeLoss:updateGradInput(input, gradOutput)
  if self.mode == 'loss' then
    if input:nElement() == self.target:nElement() then
      self.gradInput = self.crit:backward(input, self.target)
    end
    if self.normalize then
      self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
    end
    self.gradInput:mul(self.strength)
    self.gradInput:add(gradOutput)
  else
    self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  end
  return self.gradInput
end



local params = cmd:parse(arg)
main(params)
