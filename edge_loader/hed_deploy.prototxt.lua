require 'nn'
local model = {}
-- warning: module 'data [type ImageContextData]' not found
-- warning: module 'data_data_0_split [type Split]' not found
-- warning: module 'label_data_1_split [type Split]' not found
table.insert(model, {'conv1_1', nn.SpatialConvolution(3, 64, 3, 3, 1, 1, 35, 35)})
table.insert(model, {'relu1_1', nn.ReLU(true)})
table.insert(model, {'conv1_2', nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu1_2', nn.ReLU(true)})
-- warning: module 'conv1_2_relu1_2_0_split [type Split]' not found
table.insert(model, {'pool1', nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()})
table.insert(model, {'conv2_1', nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu2_1', nn.ReLU(true)})
table.insert(model, {'conv2_2', nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu2_2', nn.ReLU(true)})
-- warning: module 'conv2_2_relu2_2_0_split [type Split]' not found
table.insert(model, {'pool2', nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()})
table.insert(model, {'conv3_1', nn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu3_1', nn.ReLU(true)})
table.insert(model, {'conv3_2', nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu3_2', nn.ReLU(true)})
table.insert(model, {'conv3_3', nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu3_3', nn.ReLU(true)})
-- warning: module 'conv3_3_relu3_3_0_split [type Split]' not found
table.insert(model, {'pool3', nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()})
table.insert(model, {'conv4_1', nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu4_1', nn.ReLU(true)})
table.insert(model, {'conv4_2', nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu4_2', nn.ReLU(true)})
table.insert(model, {'conv4_3', nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu4_3', nn.ReLU(true)})
-- warning: module 'conv4_3_relu4_3_0_split [type Split]' not found
table.insert(model, {'pool4', nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()})
table.insert(model, {'conv5_1', nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu5_1', nn.ReLU(true)})
table.insert(model, {'conv5_2', nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu5_2', nn.ReLU(true)})
table.insert(model, {'conv5_3', nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu5_3', nn.ReLU(true)})
table.insert(model, {'score-dsn1', nn.SpatialConvolution(64, 1, 1, 1, 1, 1, 0, 0)})
-- warning: module 'crop [type Crop]' not found
-- warning: module 'upscore-dsn1_crop_0_split [type Split]' not found
-- warning: module ' [type SigmoidCrossEntropyLoss]' not found
table.insert(model, {'score-dsn2', nn.SpatialConvolution(128, 1, 1, 1, 1, 1, 0, 0)})
-- warning: module 'upsample_2 [type Deconvolution]' not found
-- warning: module 'crop [type Crop]' not found
-- warning: module 'upscore-dsn2_crop_0_split [type Split]' not found
-- warning: module ' [type SigmoidCrossEntropyLoss]' not found
table.insert(model, {'score-dsn3', nn.SpatialConvolution(256, 1, 1, 1, 1, 1, 0, 0)})
-- warning: module 'upsample_4 [type Deconvolution]' not found
-- warning: module 'crop [type Crop]' not found
-- warning: module 'upscore-dsn3_crop_0_split [type Split]' not found
-- warning: module ' [type SigmoidCrossEntropyLoss]' not found
table.insert(model, {'score-dsn4', nn.SpatialConvolution(512, 1, 1, 1, 1, 1, 0, 0)})
-- warning: module 'upsample_8 [type Deconvolution]' not found
-- warning: module 'crop [type Crop]' not found
-- warning: module 'upscore-dsn4_crop_0_split [type Split]' not found
-- warning: module ' [type SigmoidCrossEntropyLoss]' not found
table.insert(model, {'score-dsn5', nn.SpatialConvolution(512, 1, 1, 1, 1, 1, 0, 0)})
-- warning: module 'upsample_16 [type Deconvolution]' not found
-- warning: module 'crop [type Crop]' not found
-- warning: module 'upscore-dsn5_crop_0_split [type Split]' not found
-- warning: module ' [type SigmoidCrossEntropyLoss]' not found
-- warning: module 'concat [type Concat]' not found
table.insert(model, {'new-score-weighting', nn.SpatialConvolution(5, 1, 1, 1, 1, 1, 0, 0)})
-- warning: module ' [type SigmoidCrossEntropyLoss]' not found
return model