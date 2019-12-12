require 'torch'
require 'nn'
require 'image'

require 'fast_neural_style.EdgeLoss'

local layer_utils = require 'fast_neural_style.layer_utils'

local crit, parent = torch.class('nn.EdgeCriterion', 'nn.Criterion')


--[[
Input: args is a table with the following keys:
- cnn: A network giving the base CNN.
- content_layers: An array of layer strings
- content_weights: A list of the same length as content_layers
- style_layers: An array of layers strings
- style_weights: A list of the same length as style_layers
- deepdream_layers: Array of layer strings
- deepdream_weights: List of the same length as deepdream_layers
- loss_type: Either "L2", or "SmoothL1"
--]]
function crit:__init(args)
  
  args.edge_layers = args.edge_layers or {} 

  local next_edge_idx = 1
  local rebuilt_edgeloss_net = nn.Sequential()
  for i = 1, #args.cnn do
    if next_edge_idx <= #args.edge_layers then
      local layer = args.cnn:get(i)
      local name = layer.name      
      rebuilt_edgeloss_net:add(layer)     
      if name == 'relu5_3' then
        rebuilt_edgeloss_net:add(layer)
        next_edge_idx = next_edge_idx + 1
      end
    end
  end

  self.net = rebuilt_edgeloss_net
  self.net:evaluate()
  self.edge_loss_layers = {}

  for i, layer_string in ipairs(args.edge_layers) do
    local weight = args.edge_weights[i]
    local edge_loss_layer = nn.EdgeLoss(weight, args.loss_type)
    layer_utils.insert_after(self.net, layer_string, edge_loss_layer)
    table.insert(self.edge_loss_layers, edge_loss_layer)
  end
  
  layer_utils.trim_network(self.net)
  self.grad_net_output = torch.Tensor()
end

--[[
target: Tensor of shape (1, 3, H, W) giving pixels for style target image
--]]
function crit:setEdgeTarget(target)


  for i, edge_loss_layer in ipairs(self.edge_loss_layers) do
    edge_loss_layer:setMode('capture')
  end


  self.net:forward(target)
end


function crit:setEdgeWeight(weight)
  for i, edge_loss_layer in ipairs(self.edge_loss_layers) do
    edge_loss_layer.strength = weight
  end
end



--[[
Inputs:
- input: Tensor of shape (N, 3, H, W) giving pixels for generated images
- target: Table with the following keys:
  - content_target: Tensor of shape (N, 3, H, W)
  - style_target: Tensor of shape (1, 3, H, W)
--]]
function crit:updateOutput(input, target)
  if target.content_target then
    self:setEdgeTarget(target.content_target)
  end

  for i, edge_loss_layer in ipairs(self.edge_loss_layers) do
    edge_loss_layer:setMode('loss')
  end

  local output = self.net:forward(input)
  -- Set up a tensor of zeros to pass as gradient to net in backward pass
  self.grad_net_output:resizeAs(output):zero()
  
  self.total_edge_loss = 0
  self.edge_losses = {}
  
  for i, edge_loss_layer in ipairs(self.edge_loss_layers) do
    self.total_edge_loss = self.total_edge_loss + edge_loss_layer.loss
    table.insert(self.edge_losses, edge_loss_layer.loss)
  end
  
  self.output = self.total_edge_loss

  return self.output

end


function crit:updateGradInput(input, target)
  self.gradInput = self.net:updateGradInput(input, self.grad_net_output)
  return self.gradInput
end
