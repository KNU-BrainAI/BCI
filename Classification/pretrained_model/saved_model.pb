??0
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
?
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
DepthwiseConv2dNative

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??*
?
conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*!
shared_nameconv2d_14/kernel
}
$conv2d_14/kernel/Read/ReadVariableOpReadVariableOpconv2d_14/kernel*&
_output_shapes
:d*
dtype0
?
batch_normalization_98/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_98/gamma
?
0batch_normalization_98/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_98/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_98/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_98/beta
?
/batch_normalization_98/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_98/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_98/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_98/moving_mean
?
6batch_normalization_98/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_98/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_98/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_98/moving_variance
?
:batch_normalization_98/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_98/moving_variance*
_output_shapes
:*
dtype0
?
$depthwise_conv2d_42/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$depthwise_conv2d_42/depthwise_kernel
?
8depthwise_conv2d_42/depthwise_kernel/Read/ReadVariableOpReadVariableOp$depthwise_conv2d_42/depthwise_kernel*&
_output_shapes
:*
dtype0
?
$depthwise_conv2d_43/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$depthwise_conv2d_43/depthwise_kernel
?
8depthwise_conv2d_43/depthwise_kernel/Read/ReadVariableOpReadVariableOp$depthwise_conv2d_43/depthwise_kernel*&
_output_shapes
:*
dtype0
?
$depthwise_conv2d_44/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$depthwise_conv2d_44/depthwise_kernel
?
8depthwise_conv2d_44/depthwise_kernel/Read/ReadVariableOpReadVariableOp$depthwise_conv2d_44/depthwise_kernel*&
_output_shapes
:*
dtype0
?
batch_normalization_99/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_99/gamma
?
0batch_normalization_99/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_99/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_99/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_99/beta
?
/batch_normalization_99/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_99/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_99/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_99/moving_mean
?
6batch_normalization_99/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_99/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_99/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_99/moving_variance
?
:batch_normalization_99/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_99/moving_variance*
_output_shapes
:*
dtype0
?
batch_normalization_101/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_101/gamma
?
1batch_normalization_101/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_101/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_101/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_101/beta
?
0batch_normalization_101/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_101/beta*
_output_shapes
:*
dtype0
?
#batch_normalization_101/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_101/moving_mean
?
7batch_normalization_101/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_101/moving_mean*
_output_shapes
:*
dtype0
?
'batch_normalization_101/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_101/moving_variance
?
;batch_normalization_101/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_101/moving_variance*
_output_shapes
:*
dtype0
?
batch_normalization_103/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_103/gamma
?
1batch_normalization_103/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_103/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_103/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_103/beta
?
0batch_normalization_103/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_103/beta*
_output_shapes
:*
dtype0
?
#batch_normalization_103/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_103/moving_mean
?
7batch_normalization_103/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_103/moving_mean*
_output_shapes
:*
dtype0
?
'batch_normalization_103/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_103/moving_variance
?
;batch_normalization_103/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_103/moving_variance*
_output_shapes
:*
dtype0
?
$separable_conv2d_42/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$separable_conv2d_42/depthwise_kernel
?
8separable_conv2d_42/depthwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_42/depthwise_kernel*&
_output_shapes
: *
dtype0
?
$separable_conv2d_42/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$separable_conv2d_42/pointwise_kernel
?
8separable_conv2d_42/pointwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_42/pointwise_kernel*&
_output_shapes
:*
dtype0
?
$separable_conv2d_43/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$separable_conv2d_43/depthwise_kernel
?
8separable_conv2d_43/depthwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_43/depthwise_kernel*&
_output_shapes
:@*
dtype0
?
$separable_conv2d_43/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$separable_conv2d_43/pointwise_kernel
?
8separable_conv2d_43/pointwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_43/pointwise_kernel*&
_output_shapes
:*
dtype0
?
$separable_conv2d_44/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*5
shared_name&$separable_conv2d_44/depthwise_kernel
?
8separable_conv2d_44/depthwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_44/depthwise_kernel*'
_output_shapes
:?*
dtype0
?
$separable_conv2d_44/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$separable_conv2d_44/pointwise_kernel
?
8separable_conv2d_44/pointwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_44/pointwise_kernel*&
_output_shapes
:*
dtype0
?
batch_normalization_100/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_100/gamma
?
1batch_normalization_100/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_100/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_100/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_100/beta
?
0batch_normalization_100/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_100/beta*
_output_shapes
:*
dtype0
?
#batch_normalization_100/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_100/moving_mean
?
7batch_normalization_100/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_100/moving_mean*
_output_shapes
:*
dtype0
?
'batch_normalization_100/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_100/moving_variance
?
;batch_normalization_100/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_100/moving_variance*
_output_shapes
:*
dtype0
?
batch_normalization_102/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_102/gamma
?
1batch_normalization_102/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_102/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_102/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_102/beta
?
0batch_normalization_102/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_102/beta*
_output_shapes
:*
dtype0
?
#batch_normalization_102/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_102/moving_mean
?
7batch_normalization_102/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_102/moving_mean*
_output_shapes
:*
dtype0
?
'batch_normalization_102/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_102/moving_variance
?
;batch_normalization_102/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_102/moving_variance*
_output_shapes
:*
dtype0
?
batch_normalization_104/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_104/gamma
?
1batch_normalization_104/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_104/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_104/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_104/beta
?
0batch_normalization_104/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_104/beta*
_output_shapes
:*
dtype0
?
#batch_normalization_104/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_104/moving_mean
?
7batch_normalization_104/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_104/moving_mean*
_output_shapes
:*
dtype0
?
'batch_normalization_104/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_104/moving_variance
?
;batch_normalization_104/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_104/moving_variance*
_output_shapes
:*
dtype0
w
dense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense2/kernel
p
!dense2/kernel/Read/ReadVariableOpReadVariableOpdense2/kernel*
_output_shapes
:	?*
dtype0
n
dense2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense2/bias
g
dense2/bias/Read/ReadVariableOpReadVariableOpdense2/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/conv2d_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*(
shared_nameAdam/conv2d_14/kernel/m
?
+Adam/conv2d_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_14/kernel/m*&
_output_shapes
:d*
dtype0
?
#Adam/batch_normalization_98/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_98/gamma/m
?
7Adam/batch_normalization_98/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_98/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_98/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_98/beta/m
?
6Adam/batch_normalization_98/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_98/beta/m*
_output_shapes
:*
dtype0
?
+Adam/depthwise_conv2d_42/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+Adam/depthwise_conv2d_42/depthwise_kernel/m
?
?Adam/depthwise_conv2d_42/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp+Adam/depthwise_conv2d_42/depthwise_kernel/m*&
_output_shapes
:*
dtype0
?
+Adam/depthwise_conv2d_43/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+Adam/depthwise_conv2d_43/depthwise_kernel/m
?
?Adam/depthwise_conv2d_43/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp+Adam/depthwise_conv2d_43/depthwise_kernel/m*&
_output_shapes
:*
dtype0
?
+Adam/depthwise_conv2d_44/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+Adam/depthwise_conv2d_44/depthwise_kernel/m
?
?Adam/depthwise_conv2d_44/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp+Adam/depthwise_conv2d_44/depthwise_kernel/m*&
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_99/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_99/gamma/m
?
7Adam/batch_normalization_99/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_99/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_99/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_99/beta/m
?
6Adam/batch_normalization_99/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_99/beta/m*
_output_shapes
:*
dtype0
?
$Adam/batch_normalization_101/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_101/gamma/m
?
8Adam/batch_normalization_101/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_101/gamma/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_101/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_101/beta/m
?
7Adam/batch_normalization_101/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_101/beta/m*
_output_shapes
:*
dtype0
?
$Adam/batch_normalization_103/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_103/gamma/m
?
8Adam/batch_normalization_103/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_103/gamma/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_103/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_103/beta/m
?
7Adam/batch_normalization_103/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_103/beta/m*
_output_shapes
:*
dtype0
?
+Adam/separable_conv2d_42/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/separable_conv2d_42/depthwise_kernel/m
?
?Adam/separable_conv2d_42/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_42/depthwise_kernel/m*&
_output_shapes
: *
dtype0
?
+Adam/separable_conv2d_42/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+Adam/separable_conv2d_42/pointwise_kernel/m
?
?Adam/separable_conv2d_42/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_42/pointwise_kernel/m*&
_output_shapes
:*
dtype0
?
+Adam/separable_conv2d_43/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+Adam/separable_conv2d_43/depthwise_kernel/m
?
?Adam/separable_conv2d_43/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_43/depthwise_kernel/m*&
_output_shapes
:@*
dtype0
?
+Adam/separable_conv2d_43/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+Adam/separable_conv2d_43/pointwise_kernel/m
?
?Adam/separable_conv2d_43/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_43/pointwise_kernel/m*&
_output_shapes
:*
dtype0
?
+Adam/separable_conv2d_44/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*<
shared_name-+Adam/separable_conv2d_44/depthwise_kernel/m
?
?Adam/separable_conv2d_44/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_44/depthwise_kernel/m*'
_output_shapes
:?*
dtype0
?
+Adam/separable_conv2d_44/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+Adam/separable_conv2d_44/pointwise_kernel/m
?
?Adam/separable_conv2d_44/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_44/pointwise_kernel/m*&
_output_shapes
:*
dtype0
?
$Adam/batch_normalization_100/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_100/gamma/m
?
8Adam/batch_normalization_100/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_100/gamma/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_100/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_100/beta/m
?
7Adam/batch_normalization_100/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_100/beta/m*
_output_shapes
:*
dtype0
?
$Adam/batch_normalization_102/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_102/gamma/m
?
8Adam/batch_normalization_102/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_102/gamma/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_102/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_102/beta/m
?
7Adam/batch_normalization_102/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_102/beta/m*
_output_shapes
:*
dtype0
?
$Adam/batch_normalization_104/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_104/gamma/m
?
8Adam/batch_normalization_104/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_104/gamma/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_104/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_104/beta/m
?
7Adam/batch_normalization_104/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_104/beta/m*
_output_shapes
:*
dtype0
?
Adam/dense2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*%
shared_nameAdam/dense2/kernel/m
~
(Adam/dense2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense2/kernel/m*
_output_shapes
:	?*
dtype0
|
Adam/dense2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/dense2/bias/m
u
&Adam/dense2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense2/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*(
shared_nameAdam/conv2d_14/kernel/v
?
+Adam/conv2d_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_14/kernel/v*&
_output_shapes
:d*
dtype0
?
#Adam/batch_normalization_98/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_98/gamma/v
?
7Adam/batch_normalization_98/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_98/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_98/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_98/beta/v
?
6Adam/batch_normalization_98/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_98/beta/v*
_output_shapes
:*
dtype0
?
+Adam/depthwise_conv2d_42/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+Adam/depthwise_conv2d_42/depthwise_kernel/v
?
?Adam/depthwise_conv2d_42/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp+Adam/depthwise_conv2d_42/depthwise_kernel/v*&
_output_shapes
:*
dtype0
?
+Adam/depthwise_conv2d_43/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+Adam/depthwise_conv2d_43/depthwise_kernel/v
?
?Adam/depthwise_conv2d_43/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp+Adam/depthwise_conv2d_43/depthwise_kernel/v*&
_output_shapes
:*
dtype0
?
+Adam/depthwise_conv2d_44/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+Adam/depthwise_conv2d_44/depthwise_kernel/v
?
?Adam/depthwise_conv2d_44/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp+Adam/depthwise_conv2d_44/depthwise_kernel/v*&
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_99/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_99/gamma/v
?
7Adam/batch_normalization_99/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_99/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_99/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_99/beta/v
?
6Adam/batch_normalization_99/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_99/beta/v*
_output_shapes
:*
dtype0
?
$Adam/batch_normalization_101/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_101/gamma/v
?
8Adam/batch_normalization_101/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_101/gamma/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_101/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_101/beta/v
?
7Adam/batch_normalization_101/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_101/beta/v*
_output_shapes
:*
dtype0
?
$Adam/batch_normalization_103/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_103/gamma/v
?
8Adam/batch_normalization_103/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_103/gamma/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_103/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_103/beta/v
?
7Adam/batch_normalization_103/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_103/beta/v*
_output_shapes
:*
dtype0
?
+Adam/separable_conv2d_42/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/separable_conv2d_42/depthwise_kernel/v
?
?Adam/separable_conv2d_42/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_42/depthwise_kernel/v*&
_output_shapes
: *
dtype0
?
+Adam/separable_conv2d_42/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+Adam/separable_conv2d_42/pointwise_kernel/v
?
?Adam/separable_conv2d_42/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_42/pointwise_kernel/v*&
_output_shapes
:*
dtype0
?
+Adam/separable_conv2d_43/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+Adam/separable_conv2d_43/depthwise_kernel/v
?
?Adam/separable_conv2d_43/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_43/depthwise_kernel/v*&
_output_shapes
:@*
dtype0
?
+Adam/separable_conv2d_43/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+Adam/separable_conv2d_43/pointwise_kernel/v
?
?Adam/separable_conv2d_43/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_43/pointwise_kernel/v*&
_output_shapes
:*
dtype0
?
+Adam/separable_conv2d_44/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*<
shared_name-+Adam/separable_conv2d_44/depthwise_kernel/v
?
?Adam/separable_conv2d_44/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_44/depthwise_kernel/v*'
_output_shapes
:?*
dtype0
?
+Adam/separable_conv2d_44/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+Adam/separable_conv2d_44/pointwise_kernel/v
?
?Adam/separable_conv2d_44/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_44/pointwise_kernel/v*&
_output_shapes
:*
dtype0
?
$Adam/batch_normalization_100/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_100/gamma/v
?
8Adam/batch_normalization_100/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_100/gamma/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_100/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_100/beta/v
?
7Adam/batch_normalization_100/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_100/beta/v*
_output_shapes
:*
dtype0
?
$Adam/batch_normalization_102/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_102/gamma/v
?
8Adam/batch_normalization_102/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_102/gamma/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_102/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_102/beta/v
?
7Adam/batch_normalization_102/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_102/beta/v*
_output_shapes
:*
dtype0
?
$Adam/batch_normalization_104/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_104/gamma/v
?
8Adam/batch_normalization_104/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_104/gamma/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_104/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_104/beta/v
?
7Adam/batch_normalization_104/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_104/beta/v*
_output_shapes
:*
dtype0
?
Adam/dense2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*%
shared_nameAdam/dense2/kernel/v
~
(Adam/dense2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense2/kernel/v*
_output_shapes
:	?*
dtype0
|
Adam/dense2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/dense2/bias/v
u
&Adam/dense2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense2/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer_with_weights-8
layer-18
layer_with_weights-9
layer-19
layer_with_weights-10
layer-20
layer_with_weights-11
layer-21
layer_with_weights-12
layer-22
layer_with_weights-13
layer-23
layer-24
layer-25
layer-26
layer-27
layer-28
layer-29
layer-30
 layer-31
!layer-32
"layer-33
#layer-34
$layer-35
%layer-36
&layer_with_weights-14
&layer-37
'layer-38
(	optimizer
)trainable_variables
*regularization_losses
+	variables
,	keras_api
-
signatures
 
^

.kernel
/trainable_variables
0regularization_losses
1	variables
2	keras_api
?
3axis
	4gamma
5beta
6moving_mean
7moving_variance
8trainable_variables
9regularization_losses
:	variables
;	keras_api
h
<depthwise_kernel
=trainable_variables
>regularization_losses
?	variables
@	keras_api
h
Adepthwise_kernel
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
h
Fdepthwise_kernel
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
?
Kaxis
	Lgamma
Mbeta
Nmoving_mean
Omoving_variance
Ptrainable_variables
Qregularization_losses
R	variables
S	keras_api
?
Taxis
	Ugamma
Vbeta
Wmoving_mean
Xmoving_variance
Ytrainable_variables
Zregularization_losses
[	variables
\	keras_api
?
]axis
	^gamma
_beta
`moving_mean
amoving_variance
btrainable_variables
cregularization_losses
d	variables
e	keras_api
b
f
activation
gtrainable_variables
hregularization_losses
i	variables
j	keras_api
b
k
activation
ltrainable_variables
mregularization_losses
n	variables
o	keras_api
b
p
activation
qtrainable_variables
rregularization_losses
s	variables
t	keras_api
R
utrainable_variables
vregularization_losses
w	variables
x	keras_api
R
ytrainable_variables
zregularization_losses
{	variables
|	keras_api
S
}trainable_variables
~regularization_losses
	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
?depthwise_kernel
?pointwise_kernel
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
?depthwise_kernel
?pointwise_kernel
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
?depthwise_kernel
?pointwise_kernel
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?trainable_variables
?regularization_losses
?	variables
?	keras_api
g
?
activation
?trainable_variables
?regularization_losses
?	variables
?	keras_api
g
?
activation
?trainable_variables
?regularization_losses
?	variables
?	keras_api
g
?
activation
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
n
?kernel
	?bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate.m?4m?5m?<m?Am?Fm?Lm?Mm?Um?Vm?^m?_m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?.v?4v?5v?<v?Av?Fv?Lv?Mv?Uv?Vv?^v?_v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?
?
.0
41
52
<3
A4
F5
L6
M7
U8
V9
^10
_11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
 
?
.0
41
52
63
74
<5
A6
F7
L8
M9
N10
O11
U12
V13
W14
X15
^16
_17
`18
a19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?
?non_trainable_variables
)trainable_variables
 ?layer_regularization_losses
?metrics
*regularization_losses
+	variables
?layers
?layer_metrics
 
\Z
VARIABLE_VALUEconv2d_14/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE

.0
 

.0
?
?non_trainable_variables
/trainable_variables
 ?layer_regularization_losses
?metrics
0regularization_losses
1	variables
?layers
?layer_metrics
 
ge
VARIABLE_VALUEbatch_normalization_98/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_98/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_98/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_98/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

40
51
 

40
51
62
73
?
?non_trainable_variables
8trainable_variables
 ?layer_regularization_losses
?metrics
9regularization_losses
:	variables
?layers
?layer_metrics
zx
VARIABLE_VALUE$depthwise_conv2d_42/depthwise_kernel@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE

<0
 

<0
?
?non_trainable_variables
=trainable_variables
 ?layer_regularization_losses
?metrics
>regularization_losses
?	variables
?layers
?layer_metrics
zx
VARIABLE_VALUE$depthwise_conv2d_43/depthwise_kernel@layer_with_weights-3/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE

A0
 

A0
?
?non_trainable_variables
Btrainable_variables
 ?layer_regularization_losses
?metrics
Cregularization_losses
D	variables
?layers
?layer_metrics
zx
VARIABLE_VALUE$depthwise_conv2d_44/depthwise_kernel@layer_with_weights-4/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE

F0
 

F0
?
?non_trainable_variables
Gtrainable_variables
 ?layer_regularization_losses
?metrics
Hregularization_losses
I	variables
?layers
?layer_metrics
 
ge
VARIABLE_VALUEbatch_normalization_99/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_99/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_99/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_99/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

L0
M1
 

L0
M1
N2
O3
?
?non_trainable_variables
Ptrainable_variables
 ?layer_regularization_losses
?metrics
Qregularization_losses
R	variables
?layers
?layer_metrics
 
hf
VARIABLE_VALUEbatch_normalization_101/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_101/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#batch_normalization_101/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'batch_normalization_101/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

U0
V1
 

U0
V1
W2
X3
?
?non_trainable_variables
Ytrainable_variables
 ?layer_regularization_losses
?metrics
Zregularization_losses
[	variables
?layers
?layer_metrics
 
hf
VARIABLE_VALUEbatch_normalization_103/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_103/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#batch_normalization_103/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'batch_normalization_103/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

^0
_1
 

^0
_1
`2
a3
?
?non_trainable_variables
btrainable_variables
 ?layer_regularization_losses
?metrics
cregularization_losses
d	variables
?layers
?layer_metrics
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
 
 
 
?
?non_trainable_variables
gtrainable_variables
 ?layer_regularization_losses
?metrics
hregularization_losses
i	variables
?layers
?layer_metrics
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
 
 
 
?
?non_trainable_variables
ltrainable_variables
 ?layer_regularization_losses
?metrics
mregularization_losses
n	variables
?layers
?layer_metrics
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
 
 
 
?
?non_trainable_variables
qtrainable_variables
 ?layer_regularization_losses
?metrics
rregularization_losses
s	variables
?layers
?layer_metrics
 
 
 
?
?non_trainable_variables
utrainable_variables
 ?layer_regularization_losses
?metrics
vregularization_losses
w	variables
?layers
?layer_metrics
 
 
 
?
?non_trainable_variables
ytrainable_variables
 ?layer_regularization_losses
?metrics
zregularization_losses
{	variables
?layers
?layer_metrics
 
 
 
?
?non_trainable_variables
}trainable_variables
 ?layer_regularization_losses
?metrics
~regularization_losses
	variables
?layers
?layer_metrics
 
 
 
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
 
 
 
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
 
 
 
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
zx
VARIABLE_VALUE$separable_conv2d_42/depthwise_kernel@layer_with_weights-8/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE$separable_conv2d_42/pointwise_kernel@layer_with_weights-8/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
zx
VARIABLE_VALUE$separable_conv2d_43/depthwise_kernel@layer_with_weights-9/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE$separable_conv2d_43/pointwise_kernel@layer_with_weights-9/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
{y
VARIABLE_VALUE$separable_conv2d_44/depthwise_kernelAlayer_with_weights-10/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE$separable_conv2d_44/pointwise_kernelAlayer_with_weights-10/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
 
ig
VARIABLE_VALUEbatch_normalization_100/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEbatch_normalization_100/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE#batch_normalization_100/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE'batch_normalization_100/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 
 
?0
?1
?2
?3
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
 
ig
VARIABLE_VALUEbatch_normalization_102/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEbatch_normalization_102/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE#batch_normalization_102/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE'batch_normalization_102/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 
 
?0
?1
?2
?3
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
 
ig
VARIABLE_VALUEbatch_normalization_104/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEbatch_normalization_104/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE#batch_normalization_104/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE'batch_normalization_104/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 
 
?0
?1
?2
?3
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
 
 
 
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
 
 
 
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
 
 
 
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
 
 
 
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
 
 
 
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
 
 
 
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
 
 
 
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
 
 
 
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
 
 
 
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
 
 
 
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
 
 
 
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
 
 
 
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
 
 
 
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
ZX
VARIABLE_VALUEdense2/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense2/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
 
 
 
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
l
60
71
N2
O3
W4
X5
`6
a7
?8
?9
?10
?11
?12
?13
 

?0
?1
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
 
 
 
 
 
 

60
71
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

N0
O1
 
 
 
 

W0
X1
 
 
 
 

`0
a1
 
 
 
 
 
 
 
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
 
 
 

f0
 
 
 
 
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
 
 
 

k0
 
 
 
 
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
 
 
 

p0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 

?0
?1
 
 
 
 

?0
?1
 
 
 
 
 
 
 
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
 
 
 

?0
 
 
 
 
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
 
 
 

?0
 
 
 
 
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
 
 
 

?0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
}
VARIABLE_VALUEAdam/conv2d_14/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_98/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_98/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/depthwise_conv2d_42/depthwise_kernel/m\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/depthwise_conv2d_43/depthwise_kernel/m\layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/depthwise_conv2d_44/depthwise_kernel/m\layer_with_weights-4/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_99/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_99/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/batch_normalization_101/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_101/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/batch_normalization_103/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_103/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/separable_conv2d_42/depthwise_kernel/m\layer_with_weights-8/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/separable_conv2d_42/pointwise_kernel/m\layer_with_weights-8/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/separable_conv2d_43/depthwise_kernel/m\layer_with_weights-9/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/separable_conv2d_43/pointwise_kernel/m\layer_with_weights-9/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/separable_conv2d_44/depthwise_kernel/m]layer_with_weights-10/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/separable_conv2d_44/pointwise_kernel/m]layer_with_weights-10/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/batch_normalization_100/gamma/mRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_100/beta/mQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/batch_normalization_102/gamma/mRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_102/beta/mQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/batch_normalization_104/gamma/mRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_104/beta/mQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense2/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense2/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_14/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_98/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_98/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/depthwise_conv2d_42/depthwise_kernel/v\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/depthwise_conv2d_43/depthwise_kernel/v\layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/depthwise_conv2d_44/depthwise_kernel/v\layer_with_weights-4/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_99/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_99/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/batch_normalization_101/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_101/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/batch_normalization_103/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_103/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/separable_conv2d_42/depthwise_kernel/v\layer_with_weights-8/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/separable_conv2d_42/pointwise_kernel/v\layer_with_weights-8/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/separable_conv2d_43/depthwise_kernel/v\layer_with_weights-9/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/separable_conv2d_43/pointwise_kernel/v\layer_with_weights-9/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/separable_conv2d_44/depthwise_kernel/v]layer_with_weights-10/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/separable_conv2d_44/pointwise_kernel/v]layer_with_weights-10/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/batch_normalization_100/gamma/vRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_100/beta/vQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/batch_normalization_102/gamma/vRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_102/beta/vQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/batch_normalization_104/gamma/vRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_104/beta/vQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense2/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense2/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_15Placeholder*0
_output_shapes
:??????????*
dtype0*%
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_15conv2d_14/kernelbatch_normalization_98/gammabatch_normalization_98/beta"batch_normalization_98/moving_mean&batch_normalization_98/moving_variance$depthwise_conv2d_44/depthwise_kernel$depthwise_conv2d_43/depthwise_kernel$depthwise_conv2d_42/depthwise_kernelbatch_normalization_103/gammabatch_normalization_103/beta#batch_normalization_103/moving_mean'batch_normalization_103/moving_variancebatch_normalization_101/gammabatch_normalization_101/beta#batch_normalization_101/moving_mean'batch_normalization_101/moving_variancebatch_normalization_99/gammabatch_normalization_99/beta"batch_normalization_99/moving_mean&batch_normalization_99/moving_variance$separable_conv2d_44/depthwise_kernel$separable_conv2d_44/pointwise_kernel$separable_conv2d_43/depthwise_kernel$separable_conv2d_43/pointwise_kernel$separable_conv2d_42/depthwise_kernel$separable_conv2d_42/pointwise_kernelbatch_normalization_104/gammabatch_normalization_104/beta#batch_normalization_104/moving_mean'batch_normalization_104/moving_variancebatch_normalization_102/gammabatch_normalization_102/beta#batch_normalization_102/moving_mean'batch_normalization_102/moving_variancebatch_normalization_100/gammabatch_normalization_100/beta#batch_normalization_100/moving_mean'batch_normalization_100/moving_variancedense2/kerneldense2/bias*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*J
_read_only_resource_inputs,
*(	
 !"#$%&'(*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_586850
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?-
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_14/kernel/Read/ReadVariableOp0batch_normalization_98/gamma/Read/ReadVariableOp/batch_normalization_98/beta/Read/ReadVariableOp6batch_normalization_98/moving_mean/Read/ReadVariableOp:batch_normalization_98/moving_variance/Read/ReadVariableOp8depthwise_conv2d_42/depthwise_kernel/Read/ReadVariableOp8depthwise_conv2d_43/depthwise_kernel/Read/ReadVariableOp8depthwise_conv2d_44/depthwise_kernel/Read/ReadVariableOp0batch_normalization_99/gamma/Read/ReadVariableOp/batch_normalization_99/beta/Read/ReadVariableOp6batch_normalization_99/moving_mean/Read/ReadVariableOp:batch_normalization_99/moving_variance/Read/ReadVariableOp1batch_normalization_101/gamma/Read/ReadVariableOp0batch_normalization_101/beta/Read/ReadVariableOp7batch_normalization_101/moving_mean/Read/ReadVariableOp;batch_normalization_101/moving_variance/Read/ReadVariableOp1batch_normalization_103/gamma/Read/ReadVariableOp0batch_normalization_103/beta/Read/ReadVariableOp7batch_normalization_103/moving_mean/Read/ReadVariableOp;batch_normalization_103/moving_variance/Read/ReadVariableOp8separable_conv2d_42/depthwise_kernel/Read/ReadVariableOp8separable_conv2d_42/pointwise_kernel/Read/ReadVariableOp8separable_conv2d_43/depthwise_kernel/Read/ReadVariableOp8separable_conv2d_43/pointwise_kernel/Read/ReadVariableOp8separable_conv2d_44/depthwise_kernel/Read/ReadVariableOp8separable_conv2d_44/pointwise_kernel/Read/ReadVariableOp1batch_normalization_100/gamma/Read/ReadVariableOp0batch_normalization_100/beta/Read/ReadVariableOp7batch_normalization_100/moving_mean/Read/ReadVariableOp;batch_normalization_100/moving_variance/Read/ReadVariableOp1batch_normalization_102/gamma/Read/ReadVariableOp0batch_normalization_102/beta/Read/ReadVariableOp7batch_normalization_102/moving_mean/Read/ReadVariableOp;batch_normalization_102/moving_variance/Read/ReadVariableOp1batch_normalization_104/gamma/Read/ReadVariableOp0batch_normalization_104/beta/Read/ReadVariableOp7batch_normalization_104/moving_mean/Read/ReadVariableOp;batch_normalization_104/moving_variance/Read/ReadVariableOp!dense2/kernel/Read/ReadVariableOpdense2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_14/kernel/m/Read/ReadVariableOp7Adam/batch_normalization_98/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_98/beta/m/Read/ReadVariableOp?Adam/depthwise_conv2d_42/depthwise_kernel/m/Read/ReadVariableOp?Adam/depthwise_conv2d_43/depthwise_kernel/m/Read/ReadVariableOp?Adam/depthwise_conv2d_44/depthwise_kernel/m/Read/ReadVariableOp7Adam/batch_normalization_99/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_99/beta/m/Read/ReadVariableOp8Adam/batch_normalization_101/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_101/beta/m/Read/ReadVariableOp8Adam/batch_normalization_103/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_103/beta/m/Read/ReadVariableOp?Adam/separable_conv2d_42/depthwise_kernel/m/Read/ReadVariableOp?Adam/separable_conv2d_42/pointwise_kernel/m/Read/ReadVariableOp?Adam/separable_conv2d_43/depthwise_kernel/m/Read/ReadVariableOp?Adam/separable_conv2d_43/pointwise_kernel/m/Read/ReadVariableOp?Adam/separable_conv2d_44/depthwise_kernel/m/Read/ReadVariableOp?Adam/separable_conv2d_44/pointwise_kernel/m/Read/ReadVariableOp8Adam/batch_normalization_100/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_100/beta/m/Read/ReadVariableOp8Adam/batch_normalization_102/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_102/beta/m/Read/ReadVariableOp8Adam/batch_normalization_104/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_104/beta/m/Read/ReadVariableOp(Adam/dense2/kernel/m/Read/ReadVariableOp&Adam/dense2/bias/m/Read/ReadVariableOp+Adam/conv2d_14/kernel/v/Read/ReadVariableOp7Adam/batch_normalization_98/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_98/beta/v/Read/ReadVariableOp?Adam/depthwise_conv2d_42/depthwise_kernel/v/Read/ReadVariableOp?Adam/depthwise_conv2d_43/depthwise_kernel/v/Read/ReadVariableOp?Adam/depthwise_conv2d_44/depthwise_kernel/v/Read/ReadVariableOp7Adam/batch_normalization_99/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_99/beta/v/Read/ReadVariableOp8Adam/batch_normalization_101/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_101/beta/v/Read/ReadVariableOp8Adam/batch_normalization_103/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_103/beta/v/Read/ReadVariableOp?Adam/separable_conv2d_42/depthwise_kernel/v/Read/ReadVariableOp?Adam/separable_conv2d_42/pointwise_kernel/v/Read/ReadVariableOp?Adam/separable_conv2d_43/depthwise_kernel/v/Read/ReadVariableOp?Adam/separable_conv2d_43/pointwise_kernel/v/Read/ReadVariableOp?Adam/separable_conv2d_44/depthwise_kernel/v/Read/ReadVariableOp?Adam/separable_conv2d_44/pointwise_kernel/v/Read/ReadVariableOp8Adam/batch_normalization_100/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_100/beta/v/Read/ReadVariableOp8Adam/batch_normalization_102/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_102/beta/v/Read/ReadVariableOp8Adam/batch_normalization_104/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_104/beta/v/Read/ReadVariableOp(Adam/dense2/kernel/v/Read/ReadVariableOp&Adam/dense2/bias/v/Read/ReadVariableOpConst*r
Tink
i2g	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_589265
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_14/kernelbatch_normalization_98/gammabatch_normalization_98/beta"batch_normalization_98/moving_mean&batch_normalization_98/moving_variance$depthwise_conv2d_42/depthwise_kernel$depthwise_conv2d_43/depthwise_kernel$depthwise_conv2d_44/depthwise_kernelbatch_normalization_99/gammabatch_normalization_99/beta"batch_normalization_99/moving_mean&batch_normalization_99/moving_variancebatch_normalization_101/gammabatch_normalization_101/beta#batch_normalization_101/moving_mean'batch_normalization_101/moving_variancebatch_normalization_103/gammabatch_normalization_103/beta#batch_normalization_103/moving_mean'batch_normalization_103/moving_variance$separable_conv2d_42/depthwise_kernel$separable_conv2d_42/pointwise_kernel$separable_conv2d_43/depthwise_kernel$separable_conv2d_43/pointwise_kernel$separable_conv2d_44/depthwise_kernel$separable_conv2d_44/pointwise_kernelbatch_normalization_100/gammabatch_normalization_100/beta#batch_normalization_100/moving_mean'batch_normalization_100/moving_variancebatch_normalization_102/gammabatch_normalization_102/beta#batch_normalization_102/moving_mean'batch_normalization_102/moving_variancebatch_normalization_104/gammabatch_normalization_104/beta#batch_normalization_104/moving_mean'batch_normalization_104/moving_variancedense2/kerneldense2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_14/kernel/m#Adam/batch_normalization_98/gamma/m"Adam/batch_normalization_98/beta/m+Adam/depthwise_conv2d_42/depthwise_kernel/m+Adam/depthwise_conv2d_43/depthwise_kernel/m+Adam/depthwise_conv2d_44/depthwise_kernel/m#Adam/batch_normalization_99/gamma/m"Adam/batch_normalization_99/beta/m$Adam/batch_normalization_101/gamma/m#Adam/batch_normalization_101/beta/m$Adam/batch_normalization_103/gamma/m#Adam/batch_normalization_103/beta/m+Adam/separable_conv2d_42/depthwise_kernel/m+Adam/separable_conv2d_42/pointwise_kernel/m+Adam/separable_conv2d_43/depthwise_kernel/m+Adam/separable_conv2d_43/pointwise_kernel/m+Adam/separable_conv2d_44/depthwise_kernel/m+Adam/separable_conv2d_44/pointwise_kernel/m$Adam/batch_normalization_100/gamma/m#Adam/batch_normalization_100/beta/m$Adam/batch_normalization_102/gamma/m#Adam/batch_normalization_102/beta/m$Adam/batch_normalization_104/gamma/m#Adam/batch_normalization_104/beta/mAdam/dense2/kernel/mAdam/dense2/bias/mAdam/conv2d_14/kernel/v#Adam/batch_normalization_98/gamma/v"Adam/batch_normalization_98/beta/v+Adam/depthwise_conv2d_42/depthwise_kernel/v+Adam/depthwise_conv2d_43/depthwise_kernel/v+Adam/depthwise_conv2d_44/depthwise_kernel/v#Adam/batch_normalization_99/gamma/v"Adam/batch_normalization_99/beta/v$Adam/batch_normalization_101/gamma/v#Adam/batch_normalization_101/beta/v$Adam/batch_normalization_103/gamma/v#Adam/batch_normalization_103/beta/v+Adam/separable_conv2d_42/depthwise_kernel/v+Adam/separable_conv2d_42/pointwise_kernel/v+Adam/separable_conv2d_43/depthwise_kernel/v+Adam/separable_conv2d_43/pointwise_kernel/v+Adam/separable_conv2d_44/depthwise_kernel/v+Adam/separable_conv2d_44/pointwise_kernel/v$Adam/batch_normalization_100/gamma/v#Adam/batch_normalization_100/beta/v$Adam/batch_normalization_102/gamma/v#Adam/batch_normalization_102/beta/v$Adam/batch_normalization_104/gamma/v#Adam/batch_normalization_104/beta/vAdam/dense2/kernel/vAdam/dense2/bias/v*q
Tinj
h2f*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_589578??&
?
?
S__inference_batch_normalization_101_layer_call_and_return_conditional_losses_586008

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_103_layer_call_and_return_conditional_losses_584225

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_98_layer_call_and_return_conditional_losses_587484

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
O__inference_depthwise_conv2d_42_layer_call_and_return_conditional_losses_587572

inputs;
!depthwise_readvariableop_resource:
identity??depthwise/ReadVariableOp?
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
depthwise/Shape?
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rate?
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
	depthwisev
IdentityIdentitydepthwise:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityi
NoOpNoOp^depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
O__inference_depthwise_conv2d_44_layer_call_and_return_conditional_losses_583930

inputs;
!depthwise_readvariableop_resource:
identity??depthwise/ReadVariableOp?
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
depthwise/Shape?
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rate?
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
	depthwise?
IdentityIdentitydepthwise:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityi
NoOpNoOp^depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+???????????????????????????: 24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_103_layer_call_fn_588009

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_103_layer_call_and_return_conditional_losses_5850692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_101_layer_call_and_return_conditional_losses_587828

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
4__inference_depthwise_conv2d_44_layer_call_fn_587643

inputs!
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_depthwise_conv2d_44_layer_call_and_return_conditional_losses_5839302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+???????????????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_100_layer_call_fn_588430

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_100_layer_call_and_return_conditional_losses_5853152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????K2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????K: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_102_layer_call_and_return_conditional_losses_588461

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

?
B__inference_dense2_layer_call_and_return_conditional_losses_585429

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_102_layer_call_and_return_conditional_losses_585747

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????K:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????K2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????K: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
?
*__inference_conv2d_14_layer_call_fn_587430

inputs!
unknown:d
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_5849852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
G
+__inference_dropout_88_layer_call_fn_588188

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_88_layer_call_and_return_conditional_losses_5851772
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????K2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
Q
5__inference_average_pooling2d_84_layer_call_fn_588067

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_84_layer_call_and_return_conditional_losses_5843382
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
l
P__inference_average_pooling2d_87_layer_call_and_return_conditional_losses_588751

inputs
identity?
AvgPoolAvgPoolinputs*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
AvgPooll
IdentityIdentityAvgPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
?
O__inference_separable_conv2d_42_layer_call_and_return_conditional_losses_585238

inputsB
(separable_conv2d_readvariableop_resource: D
*separable_conv2d_readvariableop_1_resource:
identity??separable_conv2d/ReadVariableOp?!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
separable_conv2d/ReadVariableOp?
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype02#
!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2
separable_conv2d/Shape?
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rate?
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????K*
paddingSAME*
strides
2
separable_conv2d/depthwise?
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????K*
paddingVALID*
strides
2
separable_conv2d|
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*/
_output_shapes
:?????????K2

Identity?
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????K: : 2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
e
I__inference_activation_88_layer_call_and_return_conditional_losses_585138

inputs
identityi
re_lu_88/ReluReluinputs*
T0*0
_output_shapes
:??????????2
re_lu_88/Relux
IdentityIdentityre_lu_88/Relu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_102_layer_call_and_return_conditional_losses_585288

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????K:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????K2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????K: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
e
F__inference_dropout_85_layer_call_and_return_conditional_losses_585588

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
F__inference_dropout_87_layer_call_and_return_conditional_losses_588825

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
O__inference_separable_conv2d_43_layer_call_and_return_conditional_losses_588247

inputsB
(separable_conv2d_readvariableop_resource:@D
*separable_conv2d_readvariableop_1_resource:
identity??separable_conv2d/ReadVariableOp?!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02!
separable_conv2d/ReadVariableOp?
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype02#
!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"   @         2
separable_conv2d/Shape?
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rate?
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
separable_conv2d/depthwise?
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
separable_conv2d?
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_99_layer_call_and_return_conditional_losses_587668

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
Q
5__inference_average_pooling2d_86_layer_call_fn_588092

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_86_layer_call_and_return_conditional_losses_5851642
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????K2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_101_layer_call_and_return_conditional_losses_584099

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

?
O__inference_depthwise_conv2d_42_layer_call_and_return_conditional_losses_587563

inputs;
!depthwise_readvariableop_resource:
identity??depthwise/ReadVariableOp?
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
depthwise/Shape?
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rate?
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
	depthwise?
IdentityIdentitydepthwise:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityi
NoOpNoOp^depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+???????????????????????????: 24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_84_layer_call_and_return_conditional_losses_588117

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????K2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????K2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?

?
O__inference_depthwise_conv2d_44_layer_call_and_return_conditional_losses_585024

inputs;
!depthwise_readvariableop_resource:
identity??depthwise/ReadVariableOp?
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
depthwise/Shape?
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rate?
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
	depthwisev
IdentityIdentitydepthwise:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityi
NoOpNoOp^depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_104_layer_call_fn_588665

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_104_layer_call_and_return_conditional_losses_5848452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_separable_conv2d_43_layer_call_and_return_conditional_losses_588259

inputsB
(separable_conv2d_readvariableop_resource:@D
*separable_conv2d_readvariableop_1_resource:
identity??separable_conv2d/ReadVariableOp?!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02!
separable_conv2d/ReadVariableOp?
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype02#
!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"   @         2
separable_conv2d/Shape?
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rate?
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????K*
paddingSAME*
strides
2
separable_conv2d/depthwise?
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????K*
paddingVALID*
strides
2
separable_conv2d|
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*/
_output_shapes
:?????????K2

Identity?
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????K: : 2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
?
O__inference_separable_conv2d_42_layer_call_and_return_conditional_losses_588217

inputsB
(separable_conv2d_readvariableop_resource: D
*separable_conv2d_readvariableop_1_resource:
identity??separable_conv2d/ReadVariableOp?!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
separable_conv2d/ReadVariableOp?
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype02#
!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2
separable_conv2d/Shape?
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rate?
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????K*
paddingSAME*
strides
2
separable_conv2d/depthwise?
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????K*
paddingVALID*
strides
2
separable_conv2d|
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*/
_output_shapes
:?????????K2

Identity?
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????K: : 2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
??
?4
__inference__traced_save_589265
file_prefix/
+savev2_conv2d_14_kernel_read_readvariableop;
7savev2_batch_normalization_98_gamma_read_readvariableop:
6savev2_batch_normalization_98_beta_read_readvariableopA
=savev2_batch_normalization_98_moving_mean_read_readvariableopE
Asavev2_batch_normalization_98_moving_variance_read_readvariableopC
?savev2_depthwise_conv2d_42_depthwise_kernel_read_readvariableopC
?savev2_depthwise_conv2d_43_depthwise_kernel_read_readvariableopC
?savev2_depthwise_conv2d_44_depthwise_kernel_read_readvariableop;
7savev2_batch_normalization_99_gamma_read_readvariableop:
6savev2_batch_normalization_99_beta_read_readvariableopA
=savev2_batch_normalization_99_moving_mean_read_readvariableopE
Asavev2_batch_normalization_99_moving_variance_read_readvariableop<
8savev2_batch_normalization_101_gamma_read_readvariableop;
7savev2_batch_normalization_101_beta_read_readvariableopB
>savev2_batch_normalization_101_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_101_moving_variance_read_readvariableop<
8savev2_batch_normalization_103_gamma_read_readvariableop;
7savev2_batch_normalization_103_beta_read_readvariableopB
>savev2_batch_normalization_103_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_103_moving_variance_read_readvariableopC
?savev2_separable_conv2d_42_depthwise_kernel_read_readvariableopC
?savev2_separable_conv2d_42_pointwise_kernel_read_readvariableopC
?savev2_separable_conv2d_43_depthwise_kernel_read_readvariableopC
?savev2_separable_conv2d_43_pointwise_kernel_read_readvariableopC
?savev2_separable_conv2d_44_depthwise_kernel_read_readvariableopC
?savev2_separable_conv2d_44_pointwise_kernel_read_readvariableop<
8savev2_batch_normalization_100_gamma_read_readvariableop;
7savev2_batch_normalization_100_beta_read_readvariableopB
>savev2_batch_normalization_100_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_100_moving_variance_read_readvariableop<
8savev2_batch_normalization_102_gamma_read_readvariableop;
7savev2_batch_normalization_102_beta_read_readvariableopB
>savev2_batch_normalization_102_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_102_moving_variance_read_readvariableop<
8savev2_batch_normalization_104_gamma_read_readvariableop;
7savev2_batch_normalization_104_beta_read_readvariableopB
>savev2_batch_normalization_104_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_104_moving_variance_read_readvariableop,
(savev2_dense2_kernel_read_readvariableop*
&savev2_dense2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_14_kernel_m_read_readvariableopB
>savev2_adam_batch_normalization_98_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_98_beta_m_read_readvariableopJ
Fsavev2_adam_depthwise_conv2d_42_depthwise_kernel_m_read_readvariableopJ
Fsavev2_adam_depthwise_conv2d_43_depthwise_kernel_m_read_readvariableopJ
Fsavev2_adam_depthwise_conv2d_44_depthwise_kernel_m_read_readvariableopB
>savev2_adam_batch_normalization_99_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_99_beta_m_read_readvariableopC
?savev2_adam_batch_normalization_101_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_101_beta_m_read_readvariableopC
?savev2_adam_batch_normalization_103_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_103_beta_m_read_readvariableopJ
Fsavev2_adam_separable_conv2d_42_depthwise_kernel_m_read_readvariableopJ
Fsavev2_adam_separable_conv2d_42_pointwise_kernel_m_read_readvariableopJ
Fsavev2_adam_separable_conv2d_43_depthwise_kernel_m_read_readvariableopJ
Fsavev2_adam_separable_conv2d_43_pointwise_kernel_m_read_readvariableopJ
Fsavev2_adam_separable_conv2d_44_depthwise_kernel_m_read_readvariableopJ
Fsavev2_adam_separable_conv2d_44_pointwise_kernel_m_read_readvariableopC
?savev2_adam_batch_normalization_100_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_100_beta_m_read_readvariableopC
?savev2_adam_batch_normalization_102_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_102_beta_m_read_readvariableopC
?savev2_adam_batch_normalization_104_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_104_beta_m_read_readvariableop3
/savev2_adam_dense2_kernel_m_read_readvariableop1
-savev2_adam_dense2_bias_m_read_readvariableop6
2savev2_adam_conv2d_14_kernel_v_read_readvariableopB
>savev2_adam_batch_normalization_98_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_98_beta_v_read_readvariableopJ
Fsavev2_adam_depthwise_conv2d_42_depthwise_kernel_v_read_readvariableopJ
Fsavev2_adam_depthwise_conv2d_43_depthwise_kernel_v_read_readvariableopJ
Fsavev2_adam_depthwise_conv2d_44_depthwise_kernel_v_read_readvariableopB
>savev2_adam_batch_normalization_99_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_99_beta_v_read_readvariableopC
?savev2_adam_batch_normalization_101_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_101_beta_v_read_readvariableopC
?savev2_adam_batch_normalization_103_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_103_beta_v_read_readvariableopJ
Fsavev2_adam_separable_conv2d_42_depthwise_kernel_v_read_readvariableopJ
Fsavev2_adam_separable_conv2d_42_pointwise_kernel_v_read_readvariableopJ
Fsavev2_adam_separable_conv2d_43_depthwise_kernel_v_read_readvariableopJ
Fsavev2_adam_separable_conv2d_43_pointwise_kernel_v_read_readvariableopJ
Fsavev2_adam_separable_conv2d_44_depthwise_kernel_v_read_readvariableopJ
Fsavev2_adam_separable_conv2d_44_pointwise_kernel_v_read_readvariableopC
?savev2_adam_batch_normalization_100_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_100_beta_v_read_readvariableopC
?savev2_adam_batch_normalization_102_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_102_beta_v_read_readvariableopC
?savev2_adam_batch_normalization_104_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_104_beta_v_read_readvariableop3
/savev2_adam_dense2_kernel_v_read_readvariableop1
-savev2_adam_dense2_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?:
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:f*
dtype0*?:
value?:B?:fB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-3/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-4/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-8/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-8/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-9/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-9/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-10/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-10/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-4/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-8/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-8/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-10/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-10/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-4/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-8/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-8/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-10/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-10/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:f*
dtype0*?
value?B?fB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?2
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_14_kernel_read_readvariableop7savev2_batch_normalization_98_gamma_read_readvariableop6savev2_batch_normalization_98_beta_read_readvariableop=savev2_batch_normalization_98_moving_mean_read_readvariableopAsavev2_batch_normalization_98_moving_variance_read_readvariableop?savev2_depthwise_conv2d_42_depthwise_kernel_read_readvariableop?savev2_depthwise_conv2d_43_depthwise_kernel_read_readvariableop?savev2_depthwise_conv2d_44_depthwise_kernel_read_readvariableop7savev2_batch_normalization_99_gamma_read_readvariableop6savev2_batch_normalization_99_beta_read_readvariableop=savev2_batch_normalization_99_moving_mean_read_readvariableopAsavev2_batch_normalization_99_moving_variance_read_readvariableop8savev2_batch_normalization_101_gamma_read_readvariableop7savev2_batch_normalization_101_beta_read_readvariableop>savev2_batch_normalization_101_moving_mean_read_readvariableopBsavev2_batch_normalization_101_moving_variance_read_readvariableop8savev2_batch_normalization_103_gamma_read_readvariableop7savev2_batch_normalization_103_beta_read_readvariableop>savev2_batch_normalization_103_moving_mean_read_readvariableopBsavev2_batch_normalization_103_moving_variance_read_readvariableop?savev2_separable_conv2d_42_depthwise_kernel_read_readvariableop?savev2_separable_conv2d_42_pointwise_kernel_read_readvariableop?savev2_separable_conv2d_43_depthwise_kernel_read_readvariableop?savev2_separable_conv2d_43_pointwise_kernel_read_readvariableop?savev2_separable_conv2d_44_depthwise_kernel_read_readvariableop?savev2_separable_conv2d_44_pointwise_kernel_read_readvariableop8savev2_batch_normalization_100_gamma_read_readvariableop7savev2_batch_normalization_100_beta_read_readvariableop>savev2_batch_normalization_100_moving_mean_read_readvariableopBsavev2_batch_normalization_100_moving_variance_read_readvariableop8savev2_batch_normalization_102_gamma_read_readvariableop7savev2_batch_normalization_102_beta_read_readvariableop>savev2_batch_normalization_102_moving_mean_read_readvariableopBsavev2_batch_normalization_102_moving_variance_read_readvariableop8savev2_batch_normalization_104_gamma_read_readvariableop7savev2_batch_normalization_104_beta_read_readvariableop>savev2_batch_normalization_104_moving_mean_read_readvariableopBsavev2_batch_normalization_104_moving_variance_read_readvariableop(savev2_dense2_kernel_read_readvariableop&savev2_dense2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_14_kernel_m_read_readvariableop>savev2_adam_batch_normalization_98_gamma_m_read_readvariableop=savev2_adam_batch_normalization_98_beta_m_read_readvariableopFsavev2_adam_depthwise_conv2d_42_depthwise_kernel_m_read_readvariableopFsavev2_adam_depthwise_conv2d_43_depthwise_kernel_m_read_readvariableopFsavev2_adam_depthwise_conv2d_44_depthwise_kernel_m_read_readvariableop>savev2_adam_batch_normalization_99_gamma_m_read_readvariableop=savev2_adam_batch_normalization_99_beta_m_read_readvariableop?savev2_adam_batch_normalization_101_gamma_m_read_readvariableop>savev2_adam_batch_normalization_101_beta_m_read_readvariableop?savev2_adam_batch_normalization_103_gamma_m_read_readvariableop>savev2_adam_batch_normalization_103_beta_m_read_readvariableopFsavev2_adam_separable_conv2d_42_depthwise_kernel_m_read_readvariableopFsavev2_adam_separable_conv2d_42_pointwise_kernel_m_read_readvariableopFsavev2_adam_separable_conv2d_43_depthwise_kernel_m_read_readvariableopFsavev2_adam_separable_conv2d_43_pointwise_kernel_m_read_readvariableopFsavev2_adam_separable_conv2d_44_depthwise_kernel_m_read_readvariableopFsavev2_adam_separable_conv2d_44_pointwise_kernel_m_read_readvariableop?savev2_adam_batch_normalization_100_gamma_m_read_readvariableop>savev2_adam_batch_normalization_100_beta_m_read_readvariableop?savev2_adam_batch_normalization_102_gamma_m_read_readvariableop>savev2_adam_batch_normalization_102_beta_m_read_readvariableop?savev2_adam_batch_normalization_104_gamma_m_read_readvariableop>savev2_adam_batch_normalization_104_beta_m_read_readvariableop/savev2_adam_dense2_kernel_m_read_readvariableop-savev2_adam_dense2_bias_m_read_readvariableop2savev2_adam_conv2d_14_kernel_v_read_readvariableop>savev2_adam_batch_normalization_98_gamma_v_read_readvariableop=savev2_adam_batch_normalization_98_beta_v_read_readvariableopFsavev2_adam_depthwise_conv2d_42_depthwise_kernel_v_read_readvariableopFsavev2_adam_depthwise_conv2d_43_depthwise_kernel_v_read_readvariableopFsavev2_adam_depthwise_conv2d_44_depthwise_kernel_v_read_readvariableop>savev2_adam_batch_normalization_99_gamma_v_read_readvariableop=savev2_adam_batch_normalization_99_beta_v_read_readvariableop?savev2_adam_batch_normalization_101_gamma_v_read_readvariableop>savev2_adam_batch_normalization_101_beta_v_read_readvariableop?savev2_adam_batch_normalization_103_gamma_v_read_readvariableop>savev2_adam_batch_normalization_103_beta_v_read_readvariableopFsavev2_adam_separable_conv2d_42_depthwise_kernel_v_read_readvariableopFsavev2_adam_separable_conv2d_42_pointwise_kernel_v_read_readvariableopFsavev2_adam_separable_conv2d_43_depthwise_kernel_v_read_readvariableopFsavev2_adam_separable_conv2d_43_pointwise_kernel_v_read_readvariableopFsavev2_adam_separable_conv2d_44_depthwise_kernel_v_read_readvariableopFsavev2_adam_separable_conv2d_44_pointwise_kernel_v_read_readvariableop?savev2_adam_batch_normalization_100_gamma_v_read_readvariableop>savev2_adam_batch_normalization_100_beta_v_read_readvariableop?savev2_adam_batch_normalization_102_gamma_v_read_readvariableop>savev2_adam_batch_normalization_102_beta_v_read_readvariableop?savev2_adam_batch_normalization_104_gamma_v_read_readvariableop>savev2_adam_batch_normalization_104_beta_v_read_readvariableop/savev2_adam_dense2_kernel_v_read_readvariableop-savev2_adam_dense2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *t
dtypesj
h2f	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :d:::::::::::::::::::: ::@::?::::::::::::::	?:: : : : : : : : : :d:::::::::::: ::@::?::::::::	?::d:::::::::::: ::@::?::::::::	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:d: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
::,(
&
_output_shapes
::,(
&
_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: :,(
&
_output_shapes
::,(
&
_output_shapes
:@:,(
&
_output_shapes
::-)
'
_output_shapes
:?:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::  

_output_shapes
:: !

_output_shapes
:: "

_output_shapes
:: #

_output_shapes
:: $

_output_shapes
:: %

_output_shapes
:: &

_output_shapes
::%'!

_output_shapes
:	?: (

_output_shapes
::)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :,2(
&
_output_shapes
:d: 3

_output_shapes
:: 4

_output_shapes
::,5(
&
_output_shapes
::,6(
&
_output_shapes
::,7(
&
_output_shapes
:: 8

_output_shapes
:: 9

_output_shapes
:: :

_output_shapes
:: ;

_output_shapes
:: <

_output_shapes
:: =

_output_shapes
::,>(
&
_output_shapes
: :,?(
&
_output_shapes
::,@(
&
_output_shapes
:@:,A(
&
_output_shapes
::-B)
'
_output_shapes
:?:,C(
&
_output_shapes
:: D

_output_shapes
:: E

_output_shapes
:: F

_output_shapes
:: G

_output_shapes
:: H

_output_shapes
:: I

_output_shapes
::%J!

_output_shapes
:	?: K

_output_shapes
::,L(
&
_output_shapes
:d: M

_output_shapes
:: N

_output_shapes
::,O(
&
_output_shapes
::,P(
&
_output_shapes
::,Q(
&
_output_shapes
:: R

_output_shapes
:: S

_output_shapes
:: T

_output_shapes
:: U

_output_shapes
:: V

_output_shapes
:: W

_output_shapes
::,X(
&
_output_shapes
: :,Y(
&
_output_shapes
::,Z(
&
_output_shapes
:@:,[(
&
_output_shapes
::-\)
'
_output_shapes
:?:,](
&
_output_shapes
:: ^

_output_shapes
:: _

_output_shapes
:: `

_output_shapes
:: a

_output_shapes
:: b

_output_shapes
:: c

_output_shapes
::%d!

_output_shapes
:	?: e

_output_shapes
::f

_output_shapes
: 
?
l
P__inference_average_pooling2d_87_layer_call_and_return_conditional_losses_585356

inputs
identity?
AvgPoolAvgPoolinputs*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
AvgPooll
IdentityIdentityAvgPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_103_layer_call_fn_588022

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_103_layer_call_and_return_conditional_losses_5860522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
l
P__inference_average_pooling2d_89_layer_call_and_return_conditional_losses_588766

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
AvgPool?
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
O__inference_depthwise_conv2d_42_layer_call_and_return_conditional_losses_583862

inputs;
!depthwise_readvariableop_resource:
identity??depthwise/ReadVariableOp?
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
depthwise/Shape?
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rate?
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
	depthwise?
IdentityIdentitydepthwise:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityi
NoOpNoOp^depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+???????????????????????????: 24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_104_layer_call_and_return_conditional_losses_588639

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????K:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????K2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????K: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
e
I__inference_activation_88_layer_call_and_return_conditional_losses_588047

inputs
identityi
re_lu_88/ReluReluinputs*
T0*0
_output_shapes
:??????????2
re_lu_88/Relux
IdentityIdentityre_lu_88/Relu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_101_layer_call_fn_587872

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_101_layer_call_and_return_conditional_losses_5841432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
l
P__inference_average_pooling2d_88_layer_call_and_return_conditional_losses_584382

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
AvgPool?
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
O__inference_depthwise_conv2d_44_layer_call_and_return_conditional_losses_587636

inputs;
!depthwise_readvariableop_resource:
identity??depthwise/ReadVariableOp?
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
depthwise/Shape?
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rate?
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
	depthwisev
IdentityIdentitydepthwise:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityi
NoOpNoOp^depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
I__inference_activation_89_layer_call_and_return_conditional_losses_585330

inputs
identityh
re_lu_89/ReluReluinputs*
T0*/
_output_shapes
:?????????K2
re_lu_89/Reluw
IdentityIdentityre_lu_89/Relu:activations:0*
T0*/
_output_shapes
:?????????K2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_100_layer_call_and_return_conditional_losses_585703

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????K:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????K2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????K: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_103_layer_call_fn_587996

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_103_layer_call_and_return_conditional_losses_5842692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_104_layer_call_fn_588652

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_104_layer_call_and_return_conditional_losses_5848012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_84_layer_call_and_return_conditional_losses_585191

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????K2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????K2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_98_layer_call_and_return_conditional_losses_583789

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
e
F__inference_dropout_87_layer_call_and_return_conditional_losses_585611

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_98_layer_call_fn_587541

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_98_layer_call_and_return_conditional_losses_5850062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
E
)__inference_flatten1_layer_call_fn_588873

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten1_layer_call_and_return_conditional_losses_5853912
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
J
.__inference_activation_84_layer_call_fn_588032

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_84_layer_call_and_return_conditional_losses_5851522
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
F__inference_dropout_86_layer_call_and_return_conditional_losses_585872

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????K2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????K*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????K2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????K2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????K2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????K2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
_
C__inference_softmax_layer_call_and_return_conditional_losses_585440

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_103_layer_call_fn_587983

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_103_layer_call_and_return_conditional_losses_5842252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
'__inference_dense2_layer_call_fn_588929

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense2_layer_call_and_return_conditional_losses_5854292
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
O__inference_separable_conv2d_44_layer_call_and_return_conditional_losses_588289

inputsC
(separable_conv2d_readvariableop_resource:?D
*separable_conv2d_readvariableop_1_resource:
identity??separable_conv2d/ReadVariableOp?!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02!
separable_conv2d/ReadVariableOp?
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype02#
!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"   ?         2
separable_conv2d/Shape?
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rate?
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
separable_conv2d/depthwise?
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
separable_conv2d?
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_100_layer_call_and_return_conditional_losses_584549

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_flatten64_layer_call_and_return_conditional_losses_588879

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?,
D__inference_model_14_layer_call_and_return_conditional_losses_587246

inputsB
(conv2d_14_conv2d_readvariableop_resource:d<
.batch_normalization_98_readvariableop_resource:>
0batch_normalization_98_readvariableop_1_resource:M
?batch_normalization_98_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_98_fusedbatchnormv3_readvariableop_1_resource:O
5depthwise_conv2d_44_depthwise_readvariableop_resource:O
5depthwise_conv2d_43_depthwise_readvariableop_resource:O
5depthwise_conv2d_42_depthwise_readvariableop_resource:=
/batch_normalization_103_readvariableop_resource:?
1batch_normalization_103_readvariableop_1_resource:N
@batch_normalization_103_fusedbatchnormv3_readvariableop_resource:P
Bbatch_normalization_103_fusedbatchnormv3_readvariableop_1_resource:=
/batch_normalization_101_readvariableop_resource:?
1batch_normalization_101_readvariableop_1_resource:N
@batch_normalization_101_fusedbatchnormv3_readvariableop_resource:P
Bbatch_normalization_101_fusedbatchnormv3_readvariableop_1_resource:<
.batch_normalization_99_readvariableop_resource:>
0batch_normalization_99_readvariableop_1_resource:M
?batch_normalization_99_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_99_fusedbatchnormv3_readvariableop_1_resource:W
<separable_conv2d_44_separable_conv2d_readvariableop_resource:?X
>separable_conv2d_44_separable_conv2d_readvariableop_1_resource:V
<separable_conv2d_43_separable_conv2d_readvariableop_resource:@X
>separable_conv2d_43_separable_conv2d_readvariableop_1_resource:V
<separable_conv2d_42_separable_conv2d_readvariableop_resource: X
>separable_conv2d_42_separable_conv2d_readvariableop_1_resource:=
/batch_normalization_104_readvariableop_resource:?
1batch_normalization_104_readvariableop_1_resource:N
@batch_normalization_104_fusedbatchnormv3_readvariableop_resource:P
Bbatch_normalization_104_fusedbatchnormv3_readvariableop_1_resource:=
/batch_normalization_102_readvariableop_resource:?
1batch_normalization_102_readvariableop_1_resource:N
@batch_normalization_102_fusedbatchnormv3_readvariableop_resource:P
Bbatch_normalization_102_fusedbatchnormv3_readvariableop_1_resource:=
/batch_normalization_100_readvariableop_resource:?
1batch_normalization_100_readvariableop_1_resource:N
@batch_normalization_100_fusedbatchnormv3_readvariableop_resource:P
Bbatch_normalization_100_fusedbatchnormv3_readvariableop_1_resource:8
%dense2_matmul_readvariableop_resource:	?4
&dense2_biasadd_readvariableop_resource:
identity??&batch_normalization_100/AssignNewValue?(batch_normalization_100/AssignNewValue_1?7batch_normalization_100/FusedBatchNormV3/ReadVariableOp?9batch_normalization_100/FusedBatchNormV3/ReadVariableOp_1?&batch_normalization_100/ReadVariableOp?(batch_normalization_100/ReadVariableOp_1?&batch_normalization_101/AssignNewValue?(batch_normalization_101/AssignNewValue_1?7batch_normalization_101/FusedBatchNormV3/ReadVariableOp?9batch_normalization_101/FusedBatchNormV3/ReadVariableOp_1?&batch_normalization_101/ReadVariableOp?(batch_normalization_101/ReadVariableOp_1?&batch_normalization_102/AssignNewValue?(batch_normalization_102/AssignNewValue_1?7batch_normalization_102/FusedBatchNormV3/ReadVariableOp?9batch_normalization_102/FusedBatchNormV3/ReadVariableOp_1?&batch_normalization_102/ReadVariableOp?(batch_normalization_102/ReadVariableOp_1?&batch_normalization_103/AssignNewValue?(batch_normalization_103/AssignNewValue_1?7batch_normalization_103/FusedBatchNormV3/ReadVariableOp?9batch_normalization_103/FusedBatchNormV3/ReadVariableOp_1?&batch_normalization_103/ReadVariableOp?(batch_normalization_103/ReadVariableOp_1?&batch_normalization_104/AssignNewValue?(batch_normalization_104/AssignNewValue_1?7batch_normalization_104/FusedBatchNormV3/ReadVariableOp?9batch_normalization_104/FusedBatchNormV3/ReadVariableOp_1?&batch_normalization_104/ReadVariableOp?(batch_normalization_104/ReadVariableOp_1?%batch_normalization_98/AssignNewValue?'batch_normalization_98/AssignNewValue_1?6batch_normalization_98/FusedBatchNormV3/ReadVariableOp?8batch_normalization_98/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_98/ReadVariableOp?'batch_normalization_98/ReadVariableOp_1?%batch_normalization_99/AssignNewValue?'batch_normalization_99/AssignNewValue_1?6batch_normalization_99/FusedBatchNormV3/ReadVariableOp?8batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_99/ReadVariableOp?'batch_normalization_99/ReadVariableOp_1?conv2d_14/Conv2D/ReadVariableOp?dense2/BiasAdd/ReadVariableOp?dense2/MatMul/ReadVariableOp?,depthwise_conv2d_42/depthwise/ReadVariableOp?,depthwise_conv2d_43/depthwise/ReadVariableOp?,depthwise_conv2d_44/depthwise/ReadVariableOp?3separable_conv2d_42/separable_conv2d/ReadVariableOp?5separable_conv2d_42/separable_conv2d/ReadVariableOp_1?3separable_conv2d_43/separable_conv2d/ReadVariableOp?5separable_conv2d_43/separable_conv2d/ReadVariableOp_1?3separable_conv2d_44/separable_conv2d/ReadVariableOp?5separable_conv2d_44/separable_conv2d/ReadVariableOp_1?
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:d*
dtype02!
conv2d_14/Conv2D/ReadVariableOp?
conv2d_14/Conv2DConv2Dinputs'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_14/Conv2D?
%batch_normalization_98/ReadVariableOpReadVariableOp.batch_normalization_98_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_98/ReadVariableOp?
'batch_normalization_98/ReadVariableOp_1ReadVariableOp0batch_normalization_98_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_98/ReadVariableOp_1?
6batch_normalization_98/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_98_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_98/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_98/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_98_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_98/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_98/FusedBatchNormV3FusedBatchNormV3conv2d_14/Conv2D:output:0-batch_normalization_98/ReadVariableOp:value:0/batch_normalization_98/ReadVariableOp_1:value:0>batch_normalization_98/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_98/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_98/FusedBatchNormV3?
%batch_normalization_98/AssignNewValueAssignVariableOp?batch_normalization_98_fusedbatchnormv3_readvariableop_resource4batch_normalization_98/FusedBatchNormV3:batch_mean:07^batch_normalization_98/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_98/AssignNewValue?
'batch_normalization_98/AssignNewValue_1AssignVariableOpAbatch_normalization_98_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_98/FusedBatchNormV3:batch_variance:09^batch_normalization_98/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_98/AssignNewValue_1?
,depthwise_conv2d_44/depthwise/ReadVariableOpReadVariableOp5depthwise_conv2d_44_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02.
,depthwise_conv2d_44/depthwise/ReadVariableOp?
#depthwise_conv2d_44/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2%
#depthwise_conv2d_44/depthwise/Shape?
+depthwise_conv2d_44/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2-
+depthwise_conv2d_44/depthwise/dilation_rate?
depthwise_conv2d_44/depthwiseDepthwiseConv2dNativeinputs4depthwise_conv2d_44/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
depthwise_conv2d_44/depthwise?
,depthwise_conv2d_43/depthwise/ReadVariableOpReadVariableOp5depthwise_conv2d_43_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02.
,depthwise_conv2d_43/depthwise/ReadVariableOp?
#depthwise_conv2d_43/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2%
#depthwise_conv2d_43/depthwise/Shape?
+depthwise_conv2d_43/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2-
+depthwise_conv2d_43/depthwise/dilation_rate?
depthwise_conv2d_43/depthwiseDepthwiseConv2dNativeinputs4depthwise_conv2d_43/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
depthwise_conv2d_43/depthwise?
,depthwise_conv2d_42/depthwise/ReadVariableOpReadVariableOp5depthwise_conv2d_42_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02.
,depthwise_conv2d_42/depthwise/ReadVariableOp?
#depthwise_conv2d_42/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2%
#depthwise_conv2d_42/depthwise/Shape?
+depthwise_conv2d_42/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2-
+depthwise_conv2d_42/depthwise/dilation_rate?
depthwise_conv2d_42/depthwiseDepthwiseConv2dNative+batch_normalization_98/FusedBatchNormV3:y:04depthwise_conv2d_42/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
depthwise_conv2d_42/depthwise?
&batch_normalization_103/ReadVariableOpReadVariableOp/batch_normalization_103_readvariableop_resource*
_output_shapes
:*
dtype02(
&batch_normalization_103/ReadVariableOp?
(batch_normalization_103/ReadVariableOp_1ReadVariableOp1batch_normalization_103_readvariableop_1_resource*
_output_shapes
:*
dtype02*
(batch_normalization_103/ReadVariableOp_1?
7batch_normalization_103/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_103_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_103/FusedBatchNormV3/ReadVariableOp?
9batch_normalization_103/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_103_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02;
9batch_normalization_103/FusedBatchNormV3/ReadVariableOp_1?
(batch_normalization_103/FusedBatchNormV3FusedBatchNormV3&depthwise_conv2d_44/depthwise:output:0.batch_normalization_103/ReadVariableOp:value:00batch_normalization_103/ReadVariableOp_1:value:0?batch_normalization_103/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_103/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2*
(batch_normalization_103/FusedBatchNormV3?
&batch_normalization_103/AssignNewValueAssignVariableOp@batch_normalization_103_fusedbatchnormv3_readvariableop_resource5batch_normalization_103/FusedBatchNormV3:batch_mean:08^batch_normalization_103/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_103/AssignNewValue?
(batch_normalization_103/AssignNewValue_1AssignVariableOpBbatch_normalization_103_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_103/FusedBatchNormV3:batch_variance:0:^batch_normalization_103/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02*
(batch_normalization_103/AssignNewValue_1?
&batch_normalization_101/ReadVariableOpReadVariableOp/batch_normalization_101_readvariableop_resource*
_output_shapes
:*
dtype02(
&batch_normalization_101/ReadVariableOp?
(batch_normalization_101/ReadVariableOp_1ReadVariableOp1batch_normalization_101_readvariableop_1_resource*
_output_shapes
:*
dtype02*
(batch_normalization_101/ReadVariableOp_1?
7batch_normalization_101/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_101_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_101/FusedBatchNormV3/ReadVariableOp?
9batch_normalization_101/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_101_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02;
9batch_normalization_101/FusedBatchNormV3/ReadVariableOp_1?
(batch_normalization_101/FusedBatchNormV3FusedBatchNormV3&depthwise_conv2d_43/depthwise:output:0.batch_normalization_101/ReadVariableOp:value:00batch_normalization_101/ReadVariableOp_1:value:0?batch_normalization_101/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_101/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2*
(batch_normalization_101/FusedBatchNormV3?
&batch_normalization_101/AssignNewValueAssignVariableOp@batch_normalization_101_fusedbatchnormv3_readvariableop_resource5batch_normalization_101/FusedBatchNormV3:batch_mean:08^batch_normalization_101/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_101/AssignNewValue?
(batch_normalization_101/AssignNewValue_1AssignVariableOpBbatch_normalization_101_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_101/FusedBatchNormV3:batch_variance:0:^batch_normalization_101/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02*
(batch_normalization_101/AssignNewValue_1?
%batch_normalization_99/ReadVariableOpReadVariableOp.batch_normalization_99_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_99/ReadVariableOp?
'batch_normalization_99/ReadVariableOp_1ReadVariableOp0batch_normalization_99_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_99/ReadVariableOp_1?
6batch_normalization_99/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_99_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_99/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_99_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_99/FusedBatchNormV3FusedBatchNormV3&depthwise_conv2d_42/depthwise:output:0-batch_normalization_99/ReadVariableOp:value:0/batch_normalization_99/ReadVariableOp_1:value:0>batch_normalization_99/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_99/FusedBatchNormV3?
%batch_normalization_99/AssignNewValueAssignVariableOp?batch_normalization_99_fusedbatchnormv3_readvariableop_resource4batch_normalization_99/FusedBatchNormV3:batch_mean:07^batch_normalization_99/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_99/AssignNewValue?
'batch_normalization_99/AssignNewValue_1AssignVariableOpAbatch_normalization_99_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_99/FusedBatchNormV3:batch_variance:09^batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_99/AssignNewValue_1?
activation_88/re_lu_88/ReluRelu,batch_normalization_103/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
activation_88/re_lu_88/Relu?
activation_86/re_lu_86/ReluRelu,batch_normalization_101/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
activation_86/re_lu_86/Relu?
activation_84/re_lu_84/ReluRelu+batch_normalization_99/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
activation_84/re_lu_84/Relu?
average_pooling2d_88/AvgPoolAvgPool)activation_88/re_lu_88/Relu:activations:0*
T0*/
_output_shapes
:?????????K*
ksize
*
paddingVALID*
strides
2
average_pooling2d_88/AvgPool?
average_pooling2d_86/AvgPoolAvgPool)activation_86/re_lu_86/Relu:activations:0*
T0*/
_output_shapes
:?????????K*
ksize
*
paddingVALID*
strides
2
average_pooling2d_86/AvgPool?
average_pooling2d_84/AvgPoolAvgPool)activation_84/re_lu_84/Relu:activations:0*
T0*/
_output_shapes
:?????????K*
ksize
*
paddingVALID*
strides
2
average_pooling2d_84/AvgPooly
dropout_88/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_88/dropout/Const?
dropout_88/dropout/MulMul%average_pooling2d_88/AvgPool:output:0!dropout_88/dropout/Const:output:0*
T0*/
_output_shapes
:?????????K2
dropout_88/dropout/Mul?
dropout_88/dropout/ShapeShape%average_pooling2d_88/AvgPool:output:0*
T0*
_output_shapes
:2
dropout_88/dropout/Shape?
/dropout_88/dropout/random_uniform/RandomUniformRandomUniform!dropout_88/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????K*
dtype021
/dropout_88/dropout/random_uniform/RandomUniform?
!dropout_88/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_88/dropout/GreaterEqual/y?
dropout_88/dropout/GreaterEqualGreaterEqual8dropout_88/dropout/random_uniform/RandomUniform:output:0*dropout_88/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????K2!
dropout_88/dropout/GreaterEqual?
dropout_88/dropout/CastCast#dropout_88/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????K2
dropout_88/dropout/Cast?
dropout_88/dropout/Mul_1Muldropout_88/dropout/Mul:z:0dropout_88/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????K2
dropout_88/dropout/Mul_1y
dropout_86/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_86/dropout/Const?
dropout_86/dropout/MulMul%average_pooling2d_86/AvgPool:output:0!dropout_86/dropout/Const:output:0*
T0*/
_output_shapes
:?????????K2
dropout_86/dropout/Mul?
dropout_86/dropout/ShapeShape%average_pooling2d_86/AvgPool:output:0*
T0*
_output_shapes
:2
dropout_86/dropout/Shape?
/dropout_86/dropout/random_uniform/RandomUniformRandomUniform!dropout_86/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????K*
dtype021
/dropout_86/dropout/random_uniform/RandomUniform?
!dropout_86/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_86/dropout/GreaterEqual/y?
dropout_86/dropout/GreaterEqualGreaterEqual8dropout_86/dropout/random_uniform/RandomUniform:output:0*dropout_86/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????K2!
dropout_86/dropout/GreaterEqual?
dropout_86/dropout/CastCast#dropout_86/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????K2
dropout_86/dropout/Cast?
dropout_86/dropout/Mul_1Muldropout_86/dropout/Mul:z:0dropout_86/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????K2
dropout_86/dropout/Mul_1y
dropout_84/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_84/dropout/Const?
dropout_84/dropout/MulMul%average_pooling2d_84/AvgPool:output:0!dropout_84/dropout/Const:output:0*
T0*/
_output_shapes
:?????????K2
dropout_84/dropout/Mul?
dropout_84/dropout/ShapeShape%average_pooling2d_84/AvgPool:output:0*
T0*
_output_shapes
:2
dropout_84/dropout/Shape?
/dropout_84/dropout/random_uniform/RandomUniformRandomUniform!dropout_84/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????K*
dtype021
/dropout_84/dropout/random_uniform/RandomUniform?
!dropout_84/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_84/dropout/GreaterEqual/y?
dropout_84/dropout/GreaterEqualGreaterEqual8dropout_84/dropout/random_uniform/RandomUniform:output:0*dropout_84/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????K2!
dropout_84/dropout/GreaterEqual?
dropout_84/dropout/CastCast#dropout_84/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????K2
dropout_84/dropout/Cast?
dropout_84/dropout/Mul_1Muldropout_84/dropout/Mul:z:0dropout_84/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????K2
dropout_84/dropout/Mul_1?
3separable_conv2d_44/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_44_separable_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype025
3separable_conv2d_44/separable_conv2d/ReadVariableOp?
5separable_conv2d_44/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_44_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype027
5separable_conv2d_44/separable_conv2d/ReadVariableOp_1?
*separable_conv2d_44/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"   ?         2,
*separable_conv2d_44/separable_conv2d/Shape?
2separable_conv2d_44/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_44/separable_conv2d/dilation_rate?
.separable_conv2d_44/separable_conv2d/depthwiseDepthwiseConv2dNativedropout_88/dropout/Mul_1:z:0;separable_conv2d_44/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????K*
paddingSAME*
strides
20
.separable_conv2d_44/separable_conv2d/depthwise?
$separable_conv2d_44/separable_conv2dConv2D7separable_conv2d_44/separable_conv2d/depthwise:output:0=separable_conv2d_44/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????K*
paddingVALID*
strides
2&
$separable_conv2d_44/separable_conv2d?
3separable_conv2d_43/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_43_separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype025
3separable_conv2d_43/separable_conv2d/ReadVariableOp?
5separable_conv2d_43/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_43_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype027
5separable_conv2d_43/separable_conv2d/ReadVariableOp_1?
*separable_conv2d_43/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"   @         2,
*separable_conv2d_43/separable_conv2d/Shape?
2separable_conv2d_43/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_43/separable_conv2d/dilation_rate?
.separable_conv2d_43/separable_conv2d/depthwiseDepthwiseConv2dNativedropout_86/dropout/Mul_1:z:0;separable_conv2d_43/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????K*
paddingSAME*
strides
20
.separable_conv2d_43/separable_conv2d/depthwise?
$separable_conv2d_43/separable_conv2dConv2D7separable_conv2d_43/separable_conv2d/depthwise:output:0=separable_conv2d_43/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????K*
paddingVALID*
strides
2&
$separable_conv2d_43/separable_conv2d?
3separable_conv2d_42/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_42_separable_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype025
3separable_conv2d_42/separable_conv2d/ReadVariableOp?
5separable_conv2d_42/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_42_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype027
5separable_conv2d_42/separable_conv2d/ReadVariableOp_1?
*separable_conv2d_42/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*separable_conv2d_42/separable_conv2d/Shape?
2separable_conv2d_42/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_42/separable_conv2d/dilation_rate?
.separable_conv2d_42/separable_conv2d/depthwiseDepthwiseConv2dNativedropout_84/dropout/Mul_1:z:0;separable_conv2d_42/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????K*
paddingSAME*
strides
20
.separable_conv2d_42/separable_conv2d/depthwise?
$separable_conv2d_42/separable_conv2dConv2D7separable_conv2d_42/separable_conv2d/depthwise:output:0=separable_conv2d_42/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????K*
paddingVALID*
strides
2&
$separable_conv2d_42/separable_conv2d?
&batch_normalization_104/ReadVariableOpReadVariableOp/batch_normalization_104_readvariableop_resource*
_output_shapes
:*
dtype02(
&batch_normalization_104/ReadVariableOp?
(batch_normalization_104/ReadVariableOp_1ReadVariableOp1batch_normalization_104_readvariableop_1_resource*
_output_shapes
:*
dtype02*
(batch_normalization_104/ReadVariableOp_1?
7batch_normalization_104/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_104_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_104/FusedBatchNormV3/ReadVariableOp?
9batch_normalization_104/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_104_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02;
9batch_normalization_104/FusedBatchNormV3/ReadVariableOp_1?
(batch_normalization_104/FusedBatchNormV3FusedBatchNormV3-separable_conv2d_44/separable_conv2d:output:0.batch_normalization_104/ReadVariableOp:value:00batch_normalization_104/ReadVariableOp_1:value:0?batch_normalization_104/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_104/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????K:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2*
(batch_normalization_104/FusedBatchNormV3?
&batch_normalization_104/AssignNewValueAssignVariableOp@batch_normalization_104_fusedbatchnormv3_readvariableop_resource5batch_normalization_104/FusedBatchNormV3:batch_mean:08^batch_normalization_104/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_104/AssignNewValue?
(batch_normalization_104/AssignNewValue_1AssignVariableOpBbatch_normalization_104_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_104/FusedBatchNormV3:batch_variance:0:^batch_normalization_104/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02*
(batch_normalization_104/AssignNewValue_1?
&batch_normalization_102/ReadVariableOpReadVariableOp/batch_normalization_102_readvariableop_resource*
_output_shapes
:*
dtype02(
&batch_normalization_102/ReadVariableOp?
(batch_normalization_102/ReadVariableOp_1ReadVariableOp1batch_normalization_102_readvariableop_1_resource*
_output_shapes
:*
dtype02*
(batch_normalization_102/ReadVariableOp_1?
7batch_normalization_102/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_102_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_102/FusedBatchNormV3/ReadVariableOp?
9batch_normalization_102/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_102_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02;
9batch_normalization_102/FusedBatchNormV3/ReadVariableOp_1?
(batch_normalization_102/FusedBatchNormV3FusedBatchNormV3-separable_conv2d_43/separable_conv2d:output:0.batch_normalization_102/ReadVariableOp:value:00batch_normalization_102/ReadVariableOp_1:value:0?batch_normalization_102/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_102/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????K:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2*
(batch_normalization_102/FusedBatchNormV3?
&batch_normalization_102/AssignNewValueAssignVariableOp@batch_normalization_102_fusedbatchnormv3_readvariableop_resource5batch_normalization_102/FusedBatchNormV3:batch_mean:08^batch_normalization_102/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_102/AssignNewValue?
(batch_normalization_102/AssignNewValue_1AssignVariableOpBbatch_normalization_102_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_102/FusedBatchNormV3:batch_variance:0:^batch_normalization_102/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02*
(batch_normalization_102/AssignNewValue_1?
&batch_normalization_100/ReadVariableOpReadVariableOp/batch_normalization_100_readvariableop_resource*
_output_shapes
:*
dtype02(
&batch_normalization_100/ReadVariableOp?
(batch_normalization_100/ReadVariableOp_1ReadVariableOp1batch_normalization_100_readvariableop_1_resource*
_output_shapes
:*
dtype02*
(batch_normalization_100/ReadVariableOp_1?
7batch_normalization_100/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_100_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_100/FusedBatchNormV3/ReadVariableOp?
9batch_normalization_100/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_100_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02;
9batch_normalization_100/FusedBatchNormV3/ReadVariableOp_1?
(batch_normalization_100/FusedBatchNormV3FusedBatchNormV3-separable_conv2d_42/separable_conv2d:output:0.batch_normalization_100/ReadVariableOp:value:00batch_normalization_100/ReadVariableOp_1:value:0?batch_normalization_100/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_100/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????K:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2*
(batch_normalization_100/FusedBatchNormV3?
&batch_normalization_100/AssignNewValueAssignVariableOp@batch_normalization_100_fusedbatchnormv3_readvariableop_resource5batch_normalization_100/FusedBatchNormV3:batch_mean:08^batch_normalization_100/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_100/AssignNewValue?
(batch_normalization_100/AssignNewValue_1AssignVariableOpBbatch_normalization_100_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_100/FusedBatchNormV3:batch_variance:0:^batch_normalization_100/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02*
(batch_normalization_100/AssignNewValue_1?
activation_89/re_lu_89/ReluRelu,batch_normalization_104/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????K2
activation_89/re_lu_89/Relu?
activation_87/re_lu_87/ReluRelu,batch_normalization_102/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????K2
activation_87/re_lu_87/Relu?
activation_85/re_lu_85/ReluRelu,batch_normalization_100/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????K2
activation_85/re_lu_85/Relu?
average_pooling2d_89/AvgPoolAvgPool)activation_89/re_lu_89/Relu:activations:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
average_pooling2d_89/AvgPool?
average_pooling2d_87/AvgPoolAvgPool)activation_87/re_lu_87/Relu:activations:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
average_pooling2d_87/AvgPool?
average_pooling2d_85/AvgPoolAvgPool)activation_85/re_lu_85/Relu:activations:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
average_pooling2d_85/AvgPooly
dropout_89/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_89/dropout/Const?
dropout_89/dropout/MulMul%average_pooling2d_89/AvgPool:output:0!dropout_89/dropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout_89/dropout/Mul?
dropout_89/dropout/ShapeShape%average_pooling2d_89/AvgPool:output:0*
T0*
_output_shapes
:2
dropout_89/dropout/Shape?
/dropout_89/dropout/random_uniform/RandomUniformRandomUniform!dropout_89/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype021
/dropout_89/dropout/random_uniform/RandomUniform?
!dropout_89/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_89/dropout/GreaterEqual/y?
dropout_89/dropout/GreaterEqualGreaterEqual8dropout_89/dropout/random_uniform/RandomUniform:output:0*dropout_89/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2!
dropout_89/dropout/GreaterEqual?
dropout_89/dropout/CastCast#dropout_89/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout_89/dropout/Cast?
dropout_89/dropout/Mul_1Muldropout_89/dropout/Mul:z:0dropout_89/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout_89/dropout/Mul_1y
dropout_87/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_87/dropout/Const?
dropout_87/dropout/MulMul%average_pooling2d_87/AvgPool:output:0!dropout_87/dropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout_87/dropout/Mul?
dropout_87/dropout/ShapeShape%average_pooling2d_87/AvgPool:output:0*
T0*
_output_shapes
:2
dropout_87/dropout/Shape?
/dropout_87/dropout/random_uniform/RandomUniformRandomUniform!dropout_87/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype021
/dropout_87/dropout/random_uniform/RandomUniform?
!dropout_87/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_87/dropout/GreaterEqual/y?
dropout_87/dropout/GreaterEqualGreaterEqual8dropout_87/dropout/random_uniform/RandomUniform:output:0*dropout_87/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2!
dropout_87/dropout/GreaterEqual?
dropout_87/dropout/CastCast#dropout_87/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout_87/dropout/Cast?
dropout_87/dropout/Mul_1Muldropout_87/dropout/Mul:z:0dropout_87/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout_87/dropout/Mul_1y
dropout_85/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_85/dropout/Const?
dropout_85/dropout/MulMul%average_pooling2d_85/AvgPool:output:0!dropout_85/dropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout_85/dropout/Mul?
dropout_85/dropout/ShapeShape%average_pooling2d_85/AvgPool:output:0*
T0*
_output_shapes
:2
dropout_85/dropout/Shape?
/dropout_85/dropout/random_uniform/RandomUniformRandomUniform!dropout_85/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype021
/dropout_85/dropout/random_uniform/RandomUniform?
!dropout_85/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_85/dropout/GreaterEqual/y?
dropout_85/dropout/GreaterEqualGreaterEqual8dropout_85/dropout/random_uniform/RandomUniform:output:0*dropout_85/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2!
dropout_85/dropout/GreaterEqual?
dropout_85/dropout/CastCast#dropout_85/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout_85/dropout/Cast?
dropout_85/dropout/Mul_1Muldropout_85/dropout/Mul:z:0dropout_85/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout_85/dropout/Mul_1q
flatten1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
flatten1/Const?
flatten1/ReshapeReshapedropout_85/dropout/Mul_1:z:0flatten1/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten1/Reshapes
flatten64/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
flatten64/Const?
flatten64/ReshapeReshapedropout_87/dropout/Mul_1:z:0flatten64/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten64/Reshapeu
flatten128/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
flatten128/Const?
flatten128/ReshapeReshapedropout_89/dropout/Mul_1:z:0flatten128/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten128/Reshapez
concatenate_14/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_14/concat/axis?
concatenate_14/concatConcatV2flatten1/Reshape:output:0flatten64/Reshape:output:0flatten128/Reshape:output:0#concatenate_14/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatenate_14/concat?
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense2/MatMul/ReadVariableOp?
dense2/MatMulMatMulconcatenate_14/concat:output:0$dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense2/MatMul?
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense2/BiasAdd/ReadVariableOp?
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense2/BiasAddx
softmax/SoftmaxSoftmaxdense2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
softmax/Softmaxt
IdentityIdentitysoftmax/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp'^batch_normalization_100/AssignNewValue)^batch_normalization_100/AssignNewValue_18^batch_normalization_100/FusedBatchNormV3/ReadVariableOp:^batch_normalization_100/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_100/ReadVariableOp)^batch_normalization_100/ReadVariableOp_1'^batch_normalization_101/AssignNewValue)^batch_normalization_101/AssignNewValue_18^batch_normalization_101/FusedBatchNormV3/ReadVariableOp:^batch_normalization_101/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_101/ReadVariableOp)^batch_normalization_101/ReadVariableOp_1'^batch_normalization_102/AssignNewValue)^batch_normalization_102/AssignNewValue_18^batch_normalization_102/FusedBatchNormV3/ReadVariableOp:^batch_normalization_102/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_102/ReadVariableOp)^batch_normalization_102/ReadVariableOp_1'^batch_normalization_103/AssignNewValue)^batch_normalization_103/AssignNewValue_18^batch_normalization_103/FusedBatchNormV3/ReadVariableOp:^batch_normalization_103/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_103/ReadVariableOp)^batch_normalization_103/ReadVariableOp_1'^batch_normalization_104/AssignNewValue)^batch_normalization_104/AssignNewValue_18^batch_normalization_104/FusedBatchNormV3/ReadVariableOp:^batch_normalization_104/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_104/ReadVariableOp)^batch_normalization_104/ReadVariableOp_1&^batch_normalization_98/AssignNewValue(^batch_normalization_98/AssignNewValue_17^batch_normalization_98/FusedBatchNormV3/ReadVariableOp9^batch_normalization_98/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_98/ReadVariableOp(^batch_normalization_98/ReadVariableOp_1&^batch_normalization_99/AssignNewValue(^batch_normalization_99/AssignNewValue_17^batch_normalization_99/FusedBatchNormV3/ReadVariableOp9^batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_99/ReadVariableOp(^batch_normalization_99/ReadVariableOp_1 ^conv2d_14/Conv2D/ReadVariableOp^dense2/BiasAdd/ReadVariableOp^dense2/MatMul/ReadVariableOp-^depthwise_conv2d_42/depthwise/ReadVariableOp-^depthwise_conv2d_43/depthwise/ReadVariableOp-^depthwise_conv2d_44/depthwise/ReadVariableOp4^separable_conv2d_42/separable_conv2d/ReadVariableOp6^separable_conv2d_42/separable_conv2d/ReadVariableOp_14^separable_conv2d_43/separable_conv2d/ReadVariableOp6^separable_conv2d_43/separable_conv2d/ReadVariableOp_14^separable_conv2d_44/separable_conv2d/ReadVariableOp6^separable_conv2d_44/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesn
l:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_100/AssignNewValue&batch_normalization_100/AssignNewValue2T
(batch_normalization_100/AssignNewValue_1(batch_normalization_100/AssignNewValue_12r
7batch_normalization_100/FusedBatchNormV3/ReadVariableOp7batch_normalization_100/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_100/FusedBatchNormV3/ReadVariableOp_19batch_normalization_100/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_100/ReadVariableOp&batch_normalization_100/ReadVariableOp2T
(batch_normalization_100/ReadVariableOp_1(batch_normalization_100/ReadVariableOp_12P
&batch_normalization_101/AssignNewValue&batch_normalization_101/AssignNewValue2T
(batch_normalization_101/AssignNewValue_1(batch_normalization_101/AssignNewValue_12r
7batch_normalization_101/FusedBatchNormV3/ReadVariableOp7batch_normalization_101/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_101/FusedBatchNormV3/ReadVariableOp_19batch_normalization_101/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_101/ReadVariableOp&batch_normalization_101/ReadVariableOp2T
(batch_normalization_101/ReadVariableOp_1(batch_normalization_101/ReadVariableOp_12P
&batch_normalization_102/AssignNewValue&batch_normalization_102/AssignNewValue2T
(batch_normalization_102/AssignNewValue_1(batch_normalization_102/AssignNewValue_12r
7batch_normalization_102/FusedBatchNormV3/ReadVariableOp7batch_normalization_102/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_102/FusedBatchNormV3/ReadVariableOp_19batch_normalization_102/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_102/ReadVariableOp&batch_normalization_102/ReadVariableOp2T
(batch_normalization_102/ReadVariableOp_1(batch_normalization_102/ReadVariableOp_12P
&batch_normalization_103/AssignNewValue&batch_normalization_103/AssignNewValue2T
(batch_normalization_103/AssignNewValue_1(batch_normalization_103/AssignNewValue_12r
7batch_normalization_103/FusedBatchNormV3/ReadVariableOp7batch_normalization_103/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_103/FusedBatchNormV3/ReadVariableOp_19batch_normalization_103/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_103/ReadVariableOp&batch_normalization_103/ReadVariableOp2T
(batch_normalization_103/ReadVariableOp_1(batch_normalization_103/ReadVariableOp_12P
&batch_normalization_104/AssignNewValue&batch_normalization_104/AssignNewValue2T
(batch_normalization_104/AssignNewValue_1(batch_normalization_104/AssignNewValue_12r
7batch_normalization_104/FusedBatchNormV3/ReadVariableOp7batch_normalization_104/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_104/FusedBatchNormV3/ReadVariableOp_19batch_normalization_104/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_104/ReadVariableOp&batch_normalization_104/ReadVariableOp2T
(batch_normalization_104/ReadVariableOp_1(batch_normalization_104/ReadVariableOp_12N
%batch_normalization_98/AssignNewValue%batch_normalization_98/AssignNewValue2R
'batch_normalization_98/AssignNewValue_1'batch_normalization_98/AssignNewValue_12p
6batch_normalization_98/FusedBatchNormV3/ReadVariableOp6batch_normalization_98/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_98/FusedBatchNormV3/ReadVariableOp_18batch_normalization_98/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_98/ReadVariableOp%batch_normalization_98/ReadVariableOp2R
'batch_normalization_98/ReadVariableOp_1'batch_normalization_98/ReadVariableOp_12N
%batch_normalization_99/AssignNewValue%batch_normalization_99/AssignNewValue2R
'batch_normalization_99/AssignNewValue_1'batch_normalization_99/AssignNewValue_12p
6batch_normalization_99/FusedBatchNormV3/ReadVariableOp6batch_normalization_99/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_99/FusedBatchNormV3/ReadVariableOp_18batch_normalization_99/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_99/ReadVariableOp%batch_normalization_99/ReadVariableOp2R
'batch_normalization_99/ReadVariableOp_1'batch_normalization_99/ReadVariableOp_12B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2>
dense2/BiasAdd/ReadVariableOpdense2/BiasAdd/ReadVariableOp2<
dense2/MatMul/ReadVariableOpdense2/MatMul/ReadVariableOp2\
,depthwise_conv2d_42/depthwise/ReadVariableOp,depthwise_conv2d_42/depthwise/ReadVariableOp2\
,depthwise_conv2d_43/depthwise/ReadVariableOp,depthwise_conv2d_43/depthwise/ReadVariableOp2\
,depthwise_conv2d_44/depthwise/ReadVariableOp,depthwise_conv2d_44/depthwise/ReadVariableOp2j
3separable_conv2d_42/separable_conv2d/ReadVariableOp3separable_conv2d_42/separable_conv2d/ReadVariableOp2n
5separable_conv2d_42/separable_conv2d/ReadVariableOp_15separable_conv2d_42/separable_conv2d/ReadVariableOp_12j
3separable_conv2d_43/separable_conv2d/ReadVariableOp3separable_conv2d_43/separable_conv2d/ReadVariableOp2n
5separable_conv2d_43/separable_conv2d/ReadVariableOp_15separable_conv2d_43/separable_conv2d/ReadVariableOp_12j
3separable_conv2d_44/separable_conv2d/ReadVariableOp3separable_conv2d_44/separable_conv2d/ReadVariableOp2n
5separable_conv2d_44/separable_conv2d/ReadVariableOp_15separable_conv2d_44/separable_conv2d/ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_flatten64_layer_call_fn_588884

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten64_layer_call_and_return_conditional_losses_5853992
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_102_layer_call_fn_588528

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_102_layer_call_and_return_conditional_losses_5846752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
_
C__inference_softmax_layer_call_and_return_conditional_losses_588934

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
4__inference_depthwise_conv2d_42_layer_call_fn_587579

inputs!
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_depthwise_conv2d_42_layer_call_and_return_conditional_losses_5838622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+???????????????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
G
+__inference_flatten128_layer_call_fn_588895

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten128_layer_call_and_return_conditional_losses_5854072
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_103_layer_call_and_return_conditional_losses_585069

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
4__inference_depthwise_conv2d_44_layer_call_fn_587650

inputs!
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_depthwise_conv2d_44_layer_call_and_return_conditional_losses_5850242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_101_layer_call_fn_587859

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_101_layer_call_and_return_conditional_losses_5840992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
4__inference_separable_conv2d_43_layer_call_fn_588277

inputs!
unknown:@#
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_separable_conv2d_43_layer_call_and_return_conditional_losses_5852212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????K2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????K: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_98_layer_call_and_return_conditional_losses_587448

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_103_layer_call_and_return_conditional_losses_586052

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_104_layer_call_and_return_conditional_losses_588621

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????K:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????K2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????K: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_99_layer_call_and_return_conditional_losses_585964

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?	
)__inference_model_14_layer_call_fn_587416

inputs!
unknown:d
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:#
	unknown_4:#
	unknown_5:#
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:%

unknown_19:?$

unknown_20:$

unknown_21:@$

unknown_22:$

unknown_23: $

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:	?

unknown_38:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 #$'(*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_14_layer_call_and_return_conditional_losses_5863452
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesn
l:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_100_layer_call_fn_588443

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_100_layer_call_and_return_conditional_losses_5857032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????K2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????K: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
?	
)__inference_model_14_layer_call_fn_585526
input_15!
unknown:d
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:#
	unknown_4:#
	unknown_5:#
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:%

unknown_19:?$

unknown_20:$

unknown_21:@$

unknown_22:$

unknown_23: $

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:	?

unknown_38:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_15unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*J
_read_only_resource_inputs,
*(	
 !"#$%&'(*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_14_layer_call_and_return_conditional_losses_5854432
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesn
l:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:??????????
"
_user_specified_name
input_15
?	
?
8__inference_batch_normalization_102_layer_call_fn_588541

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_102_layer_call_and_return_conditional_losses_5847192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_104_layer_call_and_return_conditional_losses_588603

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_flatten128_layer_call_and_return_conditional_losses_585407

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
+__inference_dropout_85_layer_call_fn_588808

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_85_layer_call_and_return_conditional_losses_5855882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
l
P__inference_average_pooling2d_88_layer_call_and_return_conditional_losses_585158

inputs
identity?
AvgPoolAvgPoolinputs*
T0*/
_output_shapes
:?????????K*
ksize
*
paddingVALID*
strides
2	
AvgPooll
IdentityIdentityAvgPool:output:0*
T0*/
_output_shapes
:?????????K2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_101_layer_call_and_return_conditional_losses_584143

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
G
+__inference_dropout_84_layer_call_fn_588134

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_84_layer_call_and_return_conditional_losses_5851912
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????K2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
Q
5__inference_average_pooling2d_84_layer_call_fn_588072

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_84_layer_call_and_return_conditional_losses_5851702
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????K2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_100_layer_call_and_return_conditional_losses_588373

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????K:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????K2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????K: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
Q
5__inference_average_pooling2d_87_layer_call_fn_588756

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_87_layer_call_and_return_conditional_losses_5849362
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
i
/__inference_concatenate_14_layer_call_fn_588910
inputs_0
inputs_1
inputs_2
identity?
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_concatenate_14_layer_call_and_return_conditional_losses_5854172
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????:??????????:??????????:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/2
?
?
R__inference_batch_normalization_98_layer_call_and_return_conditional_losses_587466

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_88_layer_call_and_return_conditional_losses_585177

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????K2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????K2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
??
?J
"__inference__traced_restore_589578
file_prefix;
!assignvariableop_conv2d_14_kernel:d=
/assignvariableop_1_batch_normalization_98_gamma:<
.assignvariableop_2_batch_normalization_98_beta:C
5assignvariableop_3_batch_normalization_98_moving_mean:G
9assignvariableop_4_batch_normalization_98_moving_variance:Q
7assignvariableop_5_depthwise_conv2d_42_depthwise_kernel:Q
7assignvariableop_6_depthwise_conv2d_43_depthwise_kernel:Q
7assignvariableop_7_depthwise_conv2d_44_depthwise_kernel:=
/assignvariableop_8_batch_normalization_99_gamma:<
.assignvariableop_9_batch_normalization_99_beta:D
6assignvariableop_10_batch_normalization_99_moving_mean:H
:assignvariableop_11_batch_normalization_99_moving_variance:?
1assignvariableop_12_batch_normalization_101_gamma:>
0assignvariableop_13_batch_normalization_101_beta:E
7assignvariableop_14_batch_normalization_101_moving_mean:I
;assignvariableop_15_batch_normalization_101_moving_variance:?
1assignvariableop_16_batch_normalization_103_gamma:>
0assignvariableop_17_batch_normalization_103_beta:E
7assignvariableop_18_batch_normalization_103_moving_mean:I
;assignvariableop_19_batch_normalization_103_moving_variance:R
8assignvariableop_20_separable_conv2d_42_depthwise_kernel: R
8assignvariableop_21_separable_conv2d_42_pointwise_kernel:R
8assignvariableop_22_separable_conv2d_43_depthwise_kernel:@R
8assignvariableop_23_separable_conv2d_43_pointwise_kernel:S
8assignvariableop_24_separable_conv2d_44_depthwise_kernel:?R
8assignvariableop_25_separable_conv2d_44_pointwise_kernel:?
1assignvariableop_26_batch_normalization_100_gamma:>
0assignvariableop_27_batch_normalization_100_beta:E
7assignvariableop_28_batch_normalization_100_moving_mean:I
;assignvariableop_29_batch_normalization_100_moving_variance:?
1assignvariableop_30_batch_normalization_102_gamma:>
0assignvariableop_31_batch_normalization_102_beta:E
7assignvariableop_32_batch_normalization_102_moving_mean:I
;assignvariableop_33_batch_normalization_102_moving_variance:?
1assignvariableop_34_batch_normalization_104_gamma:>
0assignvariableop_35_batch_normalization_104_beta:E
7assignvariableop_36_batch_normalization_104_moving_mean:I
;assignvariableop_37_batch_normalization_104_moving_variance:4
!assignvariableop_38_dense2_kernel:	?-
assignvariableop_39_dense2_bias:'
assignvariableop_40_adam_iter:	 )
assignvariableop_41_adam_beta_1: )
assignvariableop_42_adam_beta_2: (
assignvariableop_43_adam_decay: 0
&assignvariableop_44_adam_learning_rate: #
assignvariableop_45_total: #
assignvariableop_46_count: %
assignvariableop_47_total_1: %
assignvariableop_48_count_1: E
+assignvariableop_49_adam_conv2d_14_kernel_m:dE
7assignvariableop_50_adam_batch_normalization_98_gamma_m:D
6assignvariableop_51_adam_batch_normalization_98_beta_m:Y
?assignvariableop_52_adam_depthwise_conv2d_42_depthwise_kernel_m:Y
?assignvariableop_53_adam_depthwise_conv2d_43_depthwise_kernel_m:Y
?assignvariableop_54_adam_depthwise_conv2d_44_depthwise_kernel_m:E
7assignvariableop_55_adam_batch_normalization_99_gamma_m:D
6assignvariableop_56_adam_batch_normalization_99_beta_m:F
8assignvariableop_57_adam_batch_normalization_101_gamma_m:E
7assignvariableop_58_adam_batch_normalization_101_beta_m:F
8assignvariableop_59_adam_batch_normalization_103_gamma_m:E
7assignvariableop_60_adam_batch_normalization_103_beta_m:Y
?assignvariableop_61_adam_separable_conv2d_42_depthwise_kernel_m: Y
?assignvariableop_62_adam_separable_conv2d_42_pointwise_kernel_m:Y
?assignvariableop_63_adam_separable_conv2d_43_depthwise_kernel_m:@Y
?assignvariableop_64_adam_separable_conv2d_43_pointwise_kernel_m:Z
?assignvariableop_65_adam_separable_conv2d_44_depthwise_kernel_m:?Y
?assignvariableop_66_adam_separable_conv2d_44_pointwise_kernel_m:F
8assignvariableop_67_adam_batch_normalization_100_gamma_m:E
7assignvariableop_68_adam_batch_normalization_100_beta_m:F
8assignvariableop_69_adam_batch_normalization_102_gamma_m:E
7assignvariableop_70_adam_batch_normalization_102_beta_m:F
8assignvariableop_71_adam_batch_normalization_104_gamma_m:E
7assignvariableop_72_adam_batch_normalization_104_beta_m:;
(assignvariableop_73_adam_dense2_kernel_m:	?4
&assignvariableop_74_adam_dense2_bias_m:E
+assignvariableop_75_adam_conv2d_14_kernel_v:dE
7assignvariableop_76_adam_batch_normalization_98_gamma_v:D
6assignvariableop_77_adam_batch_normalization_98_beta_v:Y
?assignvariableop_78_adam_depthwise_conv2d_42_depthwise_kernel_v:Y
?assignvariableop_79_adam_depthwise_conv2d_43_depthwise_kernel_v:Y
?assignvariableop_80_adam_depthwise_conv2d_44_depthwise_kernel_v:E
7assignvariableop_81_adam_batch_normalization_99_gamma_v:D
6assignvariableop_82_adam_batch_normalization_99_beta_v:F
8assignvariableop_83_adam_batch_normalization_101_gamma_v:E
7assignvariableop_84_adam_batch_normalization_101_beta_v:F
8assignvariableop_85_adam_batch_normalization_103_gamma_v:E
7assignvariableop_86_adam_batch_normalization_103_beta_v:Y
?assignvariableop_87_adam_separable_conv2d_42_depthwise_kernel_v: Y
?assignvariableop_88_adam_separable_conv2d_42_pointwise_kernel_v:Y
?assignvariableop_89_adam_separable_conv2d_43_depthwise_kernel_v:@Y
?assignvariableop_90_adam_separable_conv2d_43_pointwise_kernel_v:Z
?assignvariableop_91_adam_separable_conv2d_44_depthwise_kernel_v:?Y
?assignvariableop_92_adam_separable_conv2d_44_pointwise_kernel_v:F
8assignvariableop_93_adam_batch_normalization_100_gamma_v:E
7assignvariableop_94_adam_batch_normalization_100_beta_v:F
8assignvariableop_95_adam_batch_normalization_102_gamma_v:E
7assignvariableop_96_adam_batch_normalization_102_beta_v:F
8assignvariableop_97_adam_batch_normalization_104_gamma_v:E
7assignvariableop_98_adam_batch_normalization_104_beta_v:;
(assignvariableop_99_adam_dense2_kernel_v:	?5
'assignvariableop_100_adam_dense2_bias_v:
identity_102??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?;
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:f*
dtype0*?:
value?:B?:fB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-3/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-4/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-8/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-8/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-9/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-9/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-10/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-10/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-4/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-8/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-8/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-10/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-10/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-4/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-8/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-8/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-10/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-10/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:f*
dtype0*?
value?B?fB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*t
dtypesj
h2f	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_14_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp/assignvariableop_1_batch_normalization_98_gammaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_98_betaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp5assignvariableop_3_batch_normalization_98_moving_meanIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp9assignvariableop_4_batch_normalization_98_moving_varianceIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp7assignvariableop_5_depthwise_conv2d_42_depthwise_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp7assignvariableop_6_depthwise_conv2d_43_depthwise_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp7assignvariableop_7_depthwise_conv2d_44_depthwise_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_99_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_99_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_99_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_99_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp1assignvariableop_12_batch_normalization_101_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp0assignvariableop_13_batch_normalization_101_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp7assignvariableop_14_batch_normalization_101_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp;assignvariableop_15_batch_normalization_101_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp1assignvariableop_16_batch_normalization_103_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp0assignvariableop_17_batch_normalization_103_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp7assignvariableop_18_batch_normalization_103_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp;assignvariableop_19_batch_normalization_103_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp8assignvariableop_20_separable_conv2d_42_depthwise_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp8assignvariableop_21_separable_conv2d_42_pointwise_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp8assignvariableop_22_separable_conv2d_43_depthwise_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp8assignvariableop_23_separable_conv2d_43_pointwise_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp8assignvariableop_24_separable_conv2d_44_depthwise_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp8assignvariableop_25_separable_conv2d_44_pointwise_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp1assignvariableop_26_batch_normalization_100_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp0assignvariableop_27_batch_normalization_100_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp7assignvariableop_28_batch_normalization_100_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp;assignvariableop_29_batch_normalization_100_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp1assignvariableop_30_batch_normalization_102_gammaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp0assignvariableop_31_batch_normalization_102_betaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp7assignvariableop_32_batch_normalization_102_moving_meanIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp;assignvariableop_33_batch_normalization_102_moving_varianceIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp1assignvariableop_34_batch_normalization_104_gammaIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp0assignvariableop_35_batch_normalization_104_betaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp7assignvariableop_36_batch_normalization_104_moving_meanIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp;assignvariableop_37_batch_normalization_104_moving_varianceIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp!assignvariableop_38_dense2_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOpassignvariableop_39_dense2_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpassignvariableop_40_adam_iterIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpassignvariableop_41_adam_beta_1Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOpassignvariableop_42_adam_beta_2Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOpassignvariableop_43_adam_decayIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp&assignvariableop_44_adam_learning_rateIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOpassignvariableop_45_totalIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOpassignvariableop_46_countIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOpassignvariableop_47_total_1Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOpassignvariableop_48_count_1Identity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_conv2d_14_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp7assignvariableop_50_adam_batch_normalization_98_gamma_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp6assignvariableop_51_adam_batch_normalization_98_beta_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp?assignvariableop_52_adam_depthwise_conv2d_42_depthwise_kernel_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp?assignvariableop_53_adam_depthwise_conv2d_43_depthwise_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp?assignvariableop_54_adam_depthwise_conv2d_44_depthwise_kernel_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp7assignvariableop_55_adam_batch_normalization_99_gamma_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp6assignvariableop_56_adam_batch_normalization_99_beta_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp8assignvariableop_57_adam_batch_normalization_101_gamma_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp7assignvariableop_58_adam_batch_normalization_101_beta_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp8assignvariableop_59_adam_batch_normalization_103_gamma_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp7assignvariableop_60_adam_batch_normalization_103_beta_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp?assignvariableop_61_adam_separable_conv2d_42_depthwise_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp?assignvariableop_62_adam_separable_conv2d_42_pointwise_kernel_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp?assignvariableop_63_adam_separable_conv2d_43_depthwise_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp?assignvariableop_64_adam_separable_conv2d_43_pointwise_kernel_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp?assignvariableop_65_adam_separable_conv2d_44_depthwise_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp?assignvariableop_66_adam_separable_conv2d_44_pointwise_kernel_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp8assignvariableop_67_adam_batch_normalization_100_gamma_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp7assignvariableop_68_adam_batch_normalization_100_beta_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp8assignvariableop_69_adam_batch_normalization_102_gamma_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp7assignvariableop_70_adam_batch_normalization_102_beta_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp8assignvariableop_71_adam_batch_normalization_104_gamma_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp7assignvariableop_72_adam_batch_normalization_104_beta_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp(assignvariableop_73_adam_dense2_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp&assignvariableop_74_adam_dense2_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp+assignvariableop_75_adam_conv2d_14_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp7assignvariableop_76_adam_batch_normalization_98_gamma_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp6assignvariableop_77_adam_batch_normalization_98_beta_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp?assignvariableop_78_adam_depthwise_conv2d_42_depthwise_kernel_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp?assignvariableop_79_adam_depthwise_conv2d_43_depthwise_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp?assignvariableop_80_adam_depthwise_conv2d_44_depthwise_kernel_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp7assignvariableop_81_adam_batch_normalization_99_gamma_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp6assignvariableop_82_adam_batch_normalization_99_beta_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp8assignvariableop_83_adam_batch_normalization_101_gamma_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp7assignvariableop_84_adam_batch_normalization_101_beta_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp8assignvariableop_85_adam_batch_normalization_103_gamma_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp7assignvariableop_86_adam_batch_normalization_103_beta_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOp?assignvariableop_87_adam_separable_conv2d_42_depthwise_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOp?assignvariableop_88_adam_separable_conv2d_42_pointwise_kernel_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOp?assignvariableop_89_adam_separable_conv2d_43_depthwise_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOp?assignvariableop_90_adam_separable_conv2d_43_pointwise_kernel_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp?assignvariableop_91_adam_separable_conv2d_44_depthwise_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp?assignvariableop_92_adam_separable_conv2d_44_pointwise_kernel_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp8assignvariableop_93_adam_batch_normalization_100_gamma_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp7assignvariableop_94_adam_batch_normalization_100_beta_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOp8assignvariableop_95_adam_batch_normalization_102_gamma_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOp7assignvariableop_96_adam_batch_normalization_102_beta_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97?
AssignVariableOp_97AssignVariableOp8assignvariableop_97_adam_batch_normalization_104_gamma_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98?
AssignVariableOp_98AssignVariableOp7assignvariableop_98_adam_batch_normalization_104_beta_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99?
AssignVariableOp_99AssignVariableOp(assignvariableop_99_adam_dense2_kernel_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100?
AssignVariableOp_100AssignVariableOp'assignvariableop_100_adam_dense2_bias_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1009
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_101Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_101i
Identity_102IdentityIdentity_101:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_102?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"%
identity_102Identity_102:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
G
+__inference_dropout_85_layer_call_fn_588803

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_85_layer_call_and_return_conditional_losses_5853832
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_101_layer_call_and_return_conditional_losses_587792

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_100_layer_call_fn_588417

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_100_layer_call_and_return_conditional_losses_5845932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
G
+__inference_dropout_87_layer_call_fn_588830

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_87_layer_call_and_return_conditional_losses_5853762
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
l
P__inference_average_pooling2d_85_layer_call_and_return_conditional_losses_588731

inputs
identity?
AvgPoolAvgPoolinputs*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
AvgPooll
IdentityIdentityAvgPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
d
F__inference_dropout_86_layer_call_and_return_conditional_losses_585184

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????K2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????K2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
l
P__inference_average_pooling2d_86_layer_call_and_return_conditional_losses_588077

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
AvgPool?
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_101_layer_call_fn_587885

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_101_layer_call_and_return_conditional_losses_5850962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_104_layer_call_and_return_conditional_losses_588585

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_87_layer_call_and_return_conditional_losses_585376

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
l
P__inference_average_pooling2d_89_layer_call_and_return_conditional_losses_585350

inputs
identity?
AvgPoolAvgPoolinputs*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
AvgPooll
IdentityIdentityAvgPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?

?
B__inference_dense2_layer_call_and_return_conditional_losses_588920

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_100_layer_call_and_return_conditional_losses_585315

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????K:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????K2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????K: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_99_layer_call_and_return_conditional_losses_587704

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
I__inference_activation_86_layer_call_and_return_conditional_losses_588037

inputs
identityi
re_lu_86/ReluReluinputs*
T0*0
_output_shapes
:??????????2
re_lu_86/Relux
IdentityIdentityre_lu_86/Relu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
G
+__inference_dropout_89_layer_call_fn_588857

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_89_layer_call_and_return_conditional_losses_5853692
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_concatenate_14_layer_call_and_return_conditional_losses_588903
inputs_0
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????:??????????:??????????:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/2
?
e
I__inference_activation_89_layer_call_and_return_conditional_losses_588716

inputs
identityh
re_lu_89/ReluReluinputs*
T0*/
_output_shapes
:?????????K2
re_lu_89/Reluw
IdentityIdentityre_lu_89/Relu:activations:0*
T0*/
_output_shapes
:?????????K2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
e
F__inference_dropout_88_layer_call_and_return_conditional_losses_588183

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????K2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????K*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????K2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????K2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????K2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????K2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
G
+__inference_dropout_86_layer_call_fn_588161

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_86_layer_call_and_return_conditional_losses_5851842
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????K2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
J
.__inference_activation_87_layer_call_fn_588711

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_87_layer_call_and_return_conditional_losses_5853372
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????K2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
d
F__inference_dropout_86_layer_call_and_return_conditional_losses_588144

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????K2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????K2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
d
+__inference_dropout_89_layer_call_fn_588862

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_89_layer_call_and_return_conditional_losses_5856342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
Q
5__inference_average_pooling2d_89_layer_call_fn_588781

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_89_layer_call_and_return_conditional_losses_5853502
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
l
P__inference_average_pooling2d_85_layer_call_and_return_conditional_losses_584914

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
AvgPool?
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_100_layer_call_and_return_conditional_losses_588355

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_85_layer_call_and_return_conditional_losses_588786

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_99_layer_call_and_return_conditional_losses_584017

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
4__inference_separable_conv2d_42_layer_call_fn_588235

inputs!
unknown: #
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_separable_conv2d_42_layer_call_and_return_conditional_losses_5852382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????K2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????K: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
?	
)__inference_model_14_layer_call_fn_587331

inputs!
unknown:d
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:#
	unknown_4:#
	unknown_5:#
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:%

unknown_19:?$

unknown_20:$

unknown_21:@$

unknown_22:$

unknown_23: $

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:	?

unknown_38:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*J
_read_only_resource_inputs,
*(	
 !"#$%&'(*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_14_layer_call_and_return_conditional_losses_5854432
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesn
l:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
I__inference_activation_84_layer_call_and_return_conditional_losses_585152

inputs
identityi
re_lu_84/ReluReluinputs*
T0*0
_output_shapes
:??????????2
re_lu_84/Relux
IdentityIdentityre_lu_84/Relu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_99_layer_call_fn_587761

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_99_layer_call_and_return_conditional_losses_5851232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
D__inference_model_14_layer_call_and_return_conditional_losses_586345

inputs*
conv2d_14_586226:d+
batch_normalization_98_586229:+
batch_normalization_98_586231:+
batch_normalization_98_586233:+
batch_normalization_98_586235:4
depthwise_conv2d_44_586238:4
depthwise_conv2d_43_586241:4
depthwise_conv2d_42_586244:,
batch_normalization_103_586247:,
batch_normalization_103_586249:,
batch_normalization_103_586251:,
batch_normalization_103_586253:,
batch_normalization_101_586256:,
batch_normalization_101_586258:,
batch_normalization_101_586260:,
batch_normalization_101_586262:+
batch_normalization_99_586265:+
batch_normalization_99_586267:+
batch_normalization_99_586269:+
batch_normalization_99_586271:5
separable_conv2d_44_586283:?4
separable_conv2d_44_586285:4
separable_conv2d_43_586288:@4
separable_conv2d_43_586290:4
separable_conv2d_42_586293: 4
separable_conv2d_42_586295:,
batch_normalization_104_586298:,
batch_normalization_104_586300:,
batch_normalization_104_586302:,
batch_normalization_104_586304:,
batch_normalization_102_586307:,
batch_normalization_102_586309:,
batch_normalization_102_586311:,
batch_normalization_102_586313:,
batch_normalization_100_586316:,
batch_normalization_100_586318:,
batch_normalization_100_586320:,
batch_normalization_100_586322: 
dense2_586338:	?
dense2_586340:
identity??/batch_normalization_100/StatefulPartitionedCall?/batch_normalization_101/StatefulPartitionedCall?/batch_normalization_102/StatefulPartitionedCall?/batch_normalization_103/StatefulPartitionedCall?/batch_normalization_104/StatefulPartitionedCall?.batch_normalization_98/StatefulPartitionedCall?.batch_normalization_99/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?dense2/StatefulPartitionedCall?+depthwise_conv2d_42/StatefulPartitionedCall?+depthwise_conv2d_43/StatefulPartitionedCall?+depthwise_conv2d_44/StatefulPartitionedCall?"dropout_84/StatefulPartitionedCall?"dropout_85/StatefulPartitionedCall?"dropout_86/StatefulPartitionedCall?"dropout_87/StatefulPartitionedCall?"dropout_88/StatefulPartitionedCall?"dropout_89/StatefulPartitionedCall?+separable_conv2d_42/StatefulPartitionedCall?+separable_conv2d_43/StatefulPartitionedCall?+separable_conv2d_44/StatefulPartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_14_586226*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_5849852#
!conv2d_14/StatefulPartitionedCall?
.batch_normalization_98/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_98_586229batch_normalization_98_586231batch_normalization_98_586233batch_normalization_98_586235*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_98_layer_call_and_return_conditional_losses_58611720
.batch_normalization_98/StatefulPartitionedCall?
+depthwise_conv2d_44/StatefulPartitionedCallStatefulPartitionedCallinputsdepthwise_conv2d_44_586238*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_depthwise_conv2d_44_layer_call_and_return_conditional_losses_5850242-
+depthwise_conv2d_44/StatefulPartitionedCall?
+depthwise_conv2d_43/StatefulPartitionedCallStatefulPartitionedCallinputsdepthwise_conv2d_43_586241*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_depthwise_conv2d_43_layer_call_and_return_conditional_losses_5850362-
+depthwise_conv2d_43/StatefulPartitionedCall?
+depthwise_conv2d_42/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_98/StatefulPartitionedCall:output:0depthwise_conv2d_42_586244*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_depthwise_conv2d_42_layer_call_and_return_conditional_losses_5850482-
+depthwise_conv2d_42/StatefulPartitionedCall?
/batch_normalization_103/StatefulPartitionedCallStatefulPartitionedCall4depthwise_conv2d_44/StatefulPartitionedCall:output:0batch_normalization_103_586247batch_normalization_103_586249batch_normalization_103_586251batch_normalization_103_586253*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_103_layer_call_and_return_conditional_losses_58605221
/batch_normalization_103/StatefulPartitionedCall?
/batch_normalization_101/StatefulPartitionedCallStatefulPartitionedCall4depthwise_conv2d_43/StatefulPartitionedCall:output:0batch_normalization_101_586256batch_normalization_101_586258batch_normalization_101_586260batch_normalization_101_586262*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_101_layer_call_and_return_conditional_losses_58600821
/batch_normalization_101/StatefulPartitionedCall?
.batch_normalization_99/StatefulPartitionedCallStatefulPartitionedCall4depthwise_conv2d_42/StatefulPartitionedCall:output:0batch_normalization_99_586265batch_normalization_99_586267batch_normalization_99_586269batch_normalization_99_586271*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_99_layer_call_and_return_conditional_losses_58596420
.batch_normalization_99/StatefulPartitionedCall?
activation_88/PartitionedCallPartitionedCall8batch_normalization_103/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_88_layer_call_and_return_conditional_losses_5851382
activation_88/PartitionedCall?
activation_86/PartitionedCallPartitionedCall8batch_normalization_101/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_86_layer_call_and_return_conditional_losses_5851452
activation_86/PartitionedCall?
activation_84/PartitionedCallPartitionedCall7batch_normalization_99/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_84_layer_call_and_return_conditional_losses_5851522
activation_84/PartitionedCall?
$average_pooling2d_88/PartitionedCallPartitionedCall&activation_88/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_88_layer_call_and_return_conditional_losses_5851582&
$average_pooling2d_88/PartitionedCall?
$average_pooling2d_86/PartitionedCallPartitionedCall&activation_86/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_86_layer_call_and_return_conditional_losses_5851642&
$average_pooling2d_86/PartitionedCall?
$average_pooling2d_84/PartitionedCallPartitionedCall&activation_84/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_84_layer_call_and_return_conditional_losses_5851702&
$average_pooling2d_84/PartitionedCall?
"dropout_88/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_88/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_88_layer_call_and_return_conditional_losses_5858952$
"dropout_88/StatefulPartitionedCall?
"dropout_86/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_86/PartitionedCall:output:0#^dropout_88/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_86_layer_call_and_return_conditional_losses_5858722$
"dropout_86/StatefulPartitionedCall?
"dropout_84/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_84/PartitionedCall:output:0#^dropout_86/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_84_layer_call_and_return_conditional_losses_5858492$
"dropout_84/StatefulPartitionedCall?
+separable_conv2d_44/StatefulPartitionedCallStatefulPartitionedCall+dropout_88/StatefulPartitionedCall:output:0separable_conv2d_44_586283separable_conv2d_44_586285*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_separable_conv2d_44_layer_call_and_return_conditional_losses_5852042-
+separable_conv2d_44/StatefulPartitionedCall?
+separable_conv2d_43/StatefulPartitionedCallStatefulPartitionedCall+dropout_86/StatefulPartitionedCall:output:0separable_conv2d_43_586288separable_conv2d_43_586290*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_separable_conv2d_43_layer_call_and_return_conditional_losses_5852212-
+separable_conv2d_43/StatefulPartitionedCall?
+separable_conv2d_42/StatefulPartitionedCallStatefulPartitionedCall+dropout_84/StatefulPartitionedCall:output:0separable_conv2d_42_586293separable_conv2d_42_586295*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_separable_conv2d_42_layer_call_and_return_conditional_losses_5852382-
+separable_conv2d_42/StatefulPartitionedCall?
/batch_normalization_104/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_44/StatefulPartitionedCall:output:0batch_normalization_104_586298batch_normalization_104_586300batch_normalization_104_586302batch_normalization_104_586304*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_104_layer_call_and_return_conditional_losses_58579121
/batch_normalization_104/StatefulPartitionedCall?
/batch_normalization_102/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_43/StatefulPartitionedCall:output:0batch_normalization_102_586307batch_normalization_102_586309batch_normalization_102_586311batch_normalization_102_586313*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_102_layer_call_and_return_conditional_losses_58574721
/batch_normalization_102/StatefulPartitionedCall?
/batch_normalization_100/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_42/StatefulPartitionedCall:output:0batch_normalization_100_586316batch_normalization_100_586318batch_normalization_100_586320batch_normalization_100_586322*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_100_layer_call_and_return_conditional_losses_58570321
/batch_normalization_100/StatefulPartitionedCall?
activation_89/PartitionedCallPartitionedCall8batch_normalization_104/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_89_layer_call_and_return_conditional_losses_5853302
activation_89/PartitionedCall?
activation_87/PartitionedCallPartitionedCall8batch_normalization_102/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_87_layer_call_and_return_conditional_losses_5853372
activation_87/PartitionedCall?
activation_85/PartitionedCallPartitionedCall8batch_normalization_100/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_85_layer_call_and_return_conditional_losses_5853442
activation_85/PartitionedCall?
$average_pooling2d_89/PartitionedCallPartitionedCall&activation_89/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_89_layer_call_and_return_conditional_losses_5853502&
$average_pooling2d_89/PartitionedCall?
$average_pooling2d_87/PartitionedCallPartitionedCall&activation_87/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_87_layer_call_and_return_conditional_losses_5853562&
$average_pooling2d_87/PartitionedCall?
$average_pooling2d_85/PartitionedCallPartitionedCall&activation_85/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_85_layer_call_and_return_conditional_losses_5853622&
$average_pooling2d_85/PartitionedCall?
"dropout_89/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_89/PartitionedCall:output:0#^dropout_84/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_89_layer_call_and_return_conditional_losses_5856342$
"dropout_89/StatefulPartitionedCall?
"dropout_87/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_87/PartitionedCall:output:0#^dropout_89/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_87_layer_call_and_return_conditional_losses_5856112$
"dropout_87/StatefulPartitionedCall?
"dropout_85/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_85/PartitionedCall:output:0#^dropout_87/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_85_layer_call_and_return_conditional_losses_5855882$
"dropout_85/StatefulPartitionedCall?
flatten1/PartitionedCallPartitionedCall+dropout_85/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten1_layer_call_and_return_conditional_losses_5853912
flatten1/PartitionedCall?
flatten64/PartitionedCallPartitionedCall+dropout_87/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten64_layer_call_and_return_conditional_losses_5853992
flatten64/PartitionedCall?
flatten128/PartitionedCallPartitionedCall+dropout_89/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten128_layer_call_and_return_conditional_losses_5854072
flatten128/PartitionedCall?
concatenate_14/PartitionedCallPartitionedCall!flatten1/PartitionedCall:output:0"flatten64/PartitionedCall:output:0#flatten128/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_concatenate_14_layer_call_and_return_conditional_losses_5854172 
concatenate_14/PartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall'concatenate_14/PartitionedCall:output:0dense2_586338dense2_586340*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense2_layer_call_and_return_conditional_losses_5854292 
dense2/StatefulPartitionedCall?
softmax/PartitionedCallPartitionedCall'dense2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_softmax_layer_call_and_return_conditional_losses_5854402
softmax/PartitionedCall{
IdentityIdentity softmax/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp0^batch_normalization_100/StatefulPartitionedCall0^batch_normalization_101/StatefulPartitionedCall0^batch_normalization_102/StatefulPartitionedCall0^batch_normalization_103/StatefulPartitionedCall0^batch_normalization_104/StatefulPartitionedCall/^batch_normalization_98/StatefulPartitionedCall/^batch_normalization_99/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall^dense2/StatefulPartitionedCall,^depthwise_conv2d_42/StatefulPartitionedCall,^depthwise_conv2d_43/StatefulPartitionedCall,^depthwise_conv2d_44/StatefulPartitionedCall#^dropout_84/StatefulPartitionedCall#^dropout_85/StatefulPartitionedCall#^dropout_86/StatefulPartitionedCall#^dropout_87/StatefulPartitionedCall#^dropout_88/StatefulPartitionedCall#^dropout_89/StatefulPartitionedCall,^separable_conv2d_42/StatefulPartitionedCall,^separable_conv2d_43/StatefulPartitionedCall,^separable_conv2d_44/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesn
l:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_100/StatefulPartitionedCall/batch_normalization_100/StatefulPartitionedCall2b
/batch_normalization_101/StatefulPartitionedCall/batch_normalization_101/StatefulPartitionedCall2b
/batch_normalization_102/StatefulPartitionedCall/batch_normalization_102/StatefulPartitionedCall2b
/batch_normalization_103/StatefulPartitionedCall/batch_normalization_103/StatefulPartitionedCall2b
/batch_normalization_104/StatefulPartitionedCall/batch_normalization_104/StatefulPartitionedCall2`
.batch_normalization_98/StatefulPartitionedCall.batch_normalization_98/StatefulPartitionedCall2`
.batch_normalization_99/StatefulPartitionedCall.batch_normalization_99/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2Z
+depthwise_conv2d_42/StatefulPartitionedCall+depthwise_conv2d_42/StatefulPartitionedCall2Z
+depthwise_conv2d_43/StatefulPartitionedCall+depthwise_conv2d_43/StatefulPartitionedCall2Z
+depthwise_conv2d_44/StatefulPartitionedCall+depthwise_conv2d_44/StatefulPartitionedCall2H
"dropout_84/StatefulPartitionedCall"dropout_84/StatefulPartitionedCall2H
"dropout_85/StatefulPartitionedCall"dropout_85/StatefulPartitionedCall2H
"dropout_86/StatefulPartitionedCall"dropout_86/StatefulPartitionedCall2H
"dropout_87/StatefulPartitionedCall"dropout_87/StatefulPartitionedCall2H
"dropout_88/StatefulPartitionedCall"dropout_88/StatefulPartitionedCall2H
"dropout_89/StatefulPartitionedCall"dropout_89/StatefulPartitionedCall2Z
+separable_conv2d_42/StatefulPartitionedCall+separable_conv2d_42/StatefulPartitionedCall2Z
+separable_conv2d_43/StatefulPartitionedCall+separable_conv2d_43/StatefulPartitionedCall2Z
+separable_conv2d_44/StatefulPartitionedCall+separable_conv2d_44/StatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
J
.__inference_activation_86_layer_call_fn_588042

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_86_layer_call_and_return_conditional_losses_5851452
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_99_layer_call_and_return_conditional_losses_585123

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
Q
5__inference_average_pooling2d_86_layer_call_fn_588087

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_86_layer_call_and_return_conditional_losses_5843602
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
l
P__inference_average_pooling2d_86_layer_call_and_return_conditional_losses_584360

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
AvgPool?
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
O__inference_depthwise_conv2d_42_layer_call_and_return_conditional_losses_585048

inputs;
!depthwise_readvariableop_resource:
identity??depthwise/ReadVariableOp?
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
depthwise/Shape?
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rate?
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
	depthwisev
IdentityIdentitydepthwise:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityi
NoOpNoOp^depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
I__inference_activation_85_layer_call_and_return_conditional_losses_585344

inputs
identityh
re_lu_85/ReluReluinputs*
T0*/
_output_shapes
:?????????K2
re_lu_85/Reluw
IdentityIdentityre_lu_85/Relu:activations:0*
T0*/
_output_shapes
:?????????K2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_104_layer_call_and_return_conditional_losses_585261

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????K:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????K2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????K: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_99_layer_call_and_return_conditional_losses_587722

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_99_layer_call_and_return_conditional_losses_583973

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
e
I__inference_activation_87_layer_call_and_return_conditional_losses_585337

inputs
identityh
re_lu_87/ReluReluinputs*
T0*/
_output_shapes
:?????????K2
re_lu_87/Reluw
IdentityIdentityre_lu_87/Relu:activations:0*
T0*/
_output_shapes
:?????????K2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_99_layer_call_fn_587774

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_99_layer_call_and_return_conditional_losses_5859642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_101_layer_call_and_return_conditional_losses_587846

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
l
P__inference_average_pooling2d_87_layer_call_and_return_conditional_losses_588746

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
AvgPool?
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
l
P__inference_average_pooling2d_84_layer_call_and_return_conditional_losses_585170

inputs
identity?
AvgPoolAvgPoolinputs*
T0*/
_output_shapes
:?????????K*
ksize
*
paddingVALID*
strides
2	
AvgPooll
IdentityIdentityAvgPool:output:0*
T0*/
_output_shapes
:?????????K2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_14_layer_call_and_return_conditional_losses_587423

inputs8
conv2d_readvariableop_resource:d
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:d*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2Ds
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityf
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
D__inference_model_14_layer_call_and_return_conditional_losses_586635
input_15*
conv2d_14_586516:d+
batch_normalization_98_586519:+
batch_normalization_98_586521:+
batch_normalization_98_586523:+
batch_normalization_98_586525:4
depthwise_conv2d_44_586528:4
depthwise_conv2d_43_586531:4
depthwise_conv2d_42_586534:,
batch_normalization_103_586537:,
batch_normalization_103_586539:,
batch_normalization_103_586541:,
batch_normalization_103_586543:,
batch_normalization_101_586546:,
batch_normalization_101_586548:,
batch_normalization_101_586550:,
batch_normalization_101_586552:+
batch_normalization_99_586555:+
batch_normalization_99_586557:+
batch_normalization_99_586559:+
batch_normalization_99_586561:5
separable_conv2d_44_586573:?4
separable_conv2d_44_586575:4
separable_conv2d_43_586578:@4
separable_conv2d_43_586580:4
separable_conv2d_42_586583: 4
separable_conv2d_42_586585:,
batch_normalization_104_586588:,
batch_normalization_104_586590:,
batch_normalization_104_586592:,
batch_normalization_104_586594:,
batch_normalization_102_586597:,
batch_normalization_102_586599:,
batch_normalization_102_586601:,
batch_normalization_102_586603:,
batch_normalization_100_586606:,
batch_normalization_100_586608:,
batch_normalization_100_586610:,
batch_normalization_100_586612: 
dense2_586628:	?
dense2_586630:
identity??/batch_normalization_100/StatefulPartitionedCall?/batch_normalization_101/StatefulPartitionedCall?/batch_normalization_102/StatefulPartitionedCall?/batch_normalization_103/StatefulPartitionedCall?/batch_normalization_104/StatefulPartitionedCall?.batch_normalization_98/StatefulPartitionedCall?.batch_normalization_99/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?dense2/StatefulPartitionedCall?+depthwise_conv2d_42/StatefulPartitionedCall?+depthwise_conv2d_43/StatefulPartitionedCall?+depthwise_conv2d_44/StatefulPartitionedCall?+separable_conv2d_42/StatefulPartitionedCall?+separable_conv2d_43/StatefulPartitionedCall?+separable_conv2d_44/StatefulPartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCallinput_15conv2d_14_586516*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_5849852#
!conv2d_14/StatefulPartitionedCall?
.batch_normalization_98/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_98_586519batch_normalization_98_586521batch_normalization_98_586523batch_normalization_98_586525*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_98_layer_call_and_return_conditional_losses_58500620
.batch_normalization_98/StatefulPartitionedCall?
+depthwise_conv2d_44/StatefulPartitionedCallStatefulPartitionedCallinput_15depthwise_conv2d_44_586528*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_depthwise_conv2d_44_layer_call_and_return_conditional_losses_5850242-
+depthwise_conv2d_44/StatefulPartitionedCall?
+depthwise_conv2d_43/StatefulPartitionedCallStatefulPartitionedCallinput_15depthwise_conv2d_43_586531*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_depthwise_conv2d_43_layer_call_and_return_conditional_losses_5850362-
+depthwise_conv2d_43/StatefulPartitionedCall?
+depthwise_conv2d_42/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_98/StatefulPartitionedCall:output:0depthwise_conv2d_42_586534*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_depthwise_conv2d_42_layer_call_and_return_conditional_losses_5850482-
+depthwise_conv2d_42/StatefulPartitionedCall?
/batch_normalization_103/StatefulPartitionedCallStatefulPartitionedCall4depthwise_conv2d_44/StatefulPartitionedCall:output:0batch_normalization_103_586537batch_normalization_103_586539batch_normalization_103_586541batch_normalization_103_586543*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_103_layer_call_and_return_conditional_losses_58506921
/batch_normalization_103/StatefulPartitionedCall?
/batch_normalization_101/StatefulPartitionedCallStatefulPartitionedCall4depthwise_conv2d_43/StatefulPartitionedCall:output:0batch_normalization_101_586546batch_normalization_101_586548batch_normalization_101_586550batch_normalization_101_586552*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_101_layer_call_and_return_conditional_losses_58509621
/batch_normalization_101/StatefulPartitionedCall?
.batch_normalization_99/StatefulPartitionedCallStatefulPartitionedCall4depthwise_conv2d_42/StatefulPartitionedCall:output:0batch_normalization_99_586555batch_normalization_99_586557batch_normalization_99_586559batch_normalization_99_586561*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_99_layer_call_and_return_conditional_losses_58512320
.batch_normalization_99/StatefulPartitionedCall?
activation_88/PartitionedCallPartitionedCall8batch_normalization_103/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_88_layer_call_and_return_conditional_losses_5851382
activation_88/PartitionedCall?
activation_86/PartitionedCallPartitionedCall8batch_normalization_101/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_86_layer_call_and_return_conditional_losses_5851452
activation_86/PartitionedCall?
activation_84/PartitionedCallPartitionedCall7batch_normalization_99/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_84_layer_call_and_return_conditional_losses_5851522
activation_84/PartitionedCall?
$average_pooling2d_88/PartitionedCallPartitionedCall&activation_88/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_88_layer_call_and_return_conditional_losses_5851582&
$average_pooling2d_88/PartitionedCall?
$average_pooling2d_86/PartitionedCallPartitionedCall&activation_86/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_86_layer_call_and_return_conditional_losses_5851642&
$average_pooling2d_86/PartitionedCall?
$average_pooling2d_84/PartitionedCallPartitionedCall&activation_84/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_84_layer_call_and_return_conditional_losses_5851702&
$average_pooling2d_84/PartitionedCall?
dropout_88/PartitionedCallPartitionedCall-average_pooling2d_88/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_88_layer_call_and_return_conditional_losses_5851772
dropout_88/PartitionedCall?
dropout_86/PartitionedCallPartitionedCall-average_pooling2d_86/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_86_layer_call_and_return_conditional_losses_5851842
dropout_86/PartitionedCall?
dropout_84/PartitionedCallPartitionedCall-average_pooling2d_84/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_84_layer_call_and_return_conditional_losses_5851912
dropout_84/PartitionedCall?
+separable_conv2d_44/StatefulPartitionedCallStatefulPartitionedCall#dropout_88/PartitionedCall:output:0separable_conv2d_44_586573separable_conv2d_44_586575*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_separable_conv2d_44_layer_call_and_return_conditional_losses_5852042-
+separable_conv2d_44/StatefulPartitionedCall?
+separable_conv2d_43/StatefulPartitionedCallStatefulPartitionedCall#dropout_86/PartitionedCall:output:0separable_conv2d_43_586578separable_conv2d_43_586580*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_separable_conv2d_43_layer_call_and_return_conditional_losses_5852212-
+separable_conv2d_43/StatefulPartitionedCall?
+separable_conv2d_42/StatefulPartitionedCallStatefulPartitionedCall#dropout_84/PartitionedCall:output:0separable_conv2d_42_586583separable_conv2d_42_586585*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_separable_conv2d_42_layer_call_and_return_conditional_losses_5852382-
+separable_conv2d_42/StatefulPartitionedCall?
/batch_normalization_104/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_44/StatefulPartitionedCall:output:0batch_normalization_104_586588batch_normalization_104_586590batch_normalization_104_586592batch_normalization_104_586594*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_104_layer_call_and_return_conditional_losses_58526121
/batch_normalization_104/StatefulPartitionedCall?
/batch_normalization_102/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_43/StatefulPartitionedCall:output:0batch_normalization_102_586597batch_normalization_102_586599batch_normalization_102_586601batch_normalization_102_586603*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_102_layer_call_and_return_conditional_losses_58528821
/batch_normalization_102/StatefulPartitionedCall?
/batch_normalization_100/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_42/StatefulPartitionedCall:output:0batch_normalization_100_586606batch_normalization_100_586608batch_normalization_100_586610batch_normalization_100_586612*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_100_layer_call_and_return_conditional_losses_58531521
/batch_normalization_100/StatefulPartitionedCall?
activation_89/PartitionedCallPartitionedCall8batch_normalization_104/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_89_layer_call_and_return_conditional_losses_5853302
activation_89/PartitionedCall?
activation_87/PartitionedCallPartitionedCall8batch_normalization_102/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_87_layer_call_and_return_conditional_losses_5853372
activation_87/PartitionedCall?
activation_85/PartitionedCallPartitionedCall8batch_normalization_100/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_85_layer_call_and_return_conditional_losses_5853442
activation_85/PartitionedCall?
$average_pooling2d_89/PartitionedCallPartitionedCall&activation_89/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_89_layer_call_and_return_conditional_losses_5853502&
$average_pooling2d_89/PartitionedCall?
$average_pooling2d_87/PartitionedCallPartitionedCall&activation_87/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_87_layer_call_and_return_conditional_losses_5853562&
$average_pooling2d_87/PartitionedCall?
$average_pooling2d_85/PartitionedCallPartitionedCall&activation_85/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_85_layer_call_and_return_conditional_losses_5853622&
$average_pooling2d_85/PartitionedCall?
dropout_89/PartitionedCallPartitionedCall-average_pooling2d_89/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_89_layer_call_and_return_conditional_losses_5853692
dropout_89/PartitionedCall?
dropout_87/PartitionedCallPartitionedCall-average_pooling2d_87/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_87_layer_call_and_return_conditional_losses_5853762
dropout_87/PartitionedCall?
dropout_85/PartitionedCallPartitionedCall-average_pooling2d_85/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_85_layer_call_and_return_conditional_losses_5853832
dropout_85/PartitionedCall?
flatten1/PartitionedCallPartitionedCall#dropout_85/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten1_layer_call_and_return_conditional_losses_5853912
flatten1/PartitionedCall?
flatten64/PartitionedCallPartitionedCall#dropout_87/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten64_layer_call_and_return_conditional_losses_5853992
flatten64/PartitionedCall?
flatten128/PartitionedCallPartitionedCall#dropout_89/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten128_layer_call_and_return_conditional_losses_5854072
flatten128/PartitionedCall?
concatenate_14/PartitionedCallPartitionedCall!flatten1/PartitionedCall:output:0"flatten64/PartitionedCall:output:0#flatten128/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_concatenate_14_layer_call_and_return_conditional_losses_5854172 
concatenate_14/PartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall'concatenate_14/PartitionedCall:output:0dense2_586628dense2_586630*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense2_layer_call_and_return_conditional_losses_5854292 
dense2/StatefulPartitionedCall?
softmax/PartitionedCallPartitionedCall'dense2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_softmax_layer_call_and_return_conditional_losses_5854402
softmax/PartitionedCall{
IdentityIdentity softmax/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp0^batch_normalization_100/StatefulPartitionedCall0^batch_normalization_101/StatefulPartitionedCall0^batch_normalization_102/StatefulPartitionedCall0^batch_normalization_103/StatefulPartitionedCall0^batch_normalization_104/StatefulPartitionedCall/^batch_normalization_98/StatefulPartitionedCall/^batch_normalization_99/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall^dense2/StatefulPartitionedCall,^depthwise_conv2d_42/StatefulPartitionedCall,^depthwise_conv2d_43/StatefulPartitionedCall,^depthwise_conv2d_44/StatefulPartitionedCall,^separable_conv2d_42/StatefulPartitionedCall,^separable_conv2d_43/StatefulPartitionedCall,^separable_conv2d_44/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesn
l:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_100/StatefulPartitionedCall/batch_normalization_100/StatefulPartitionedCall2b
/batch_normalization_101/StatefulPartitionedCall/batch_normalization_101/StatefulPartitionedCall2b
/batch_normalization_102/StatefulPartitionedCall/batch_normalization_102/StatefulPartitionedCall2b
/batch_normalization_103/StatefulPartitionedCall/batch_normalization_103/StatefulPartitionedCall2b
/batch_normalization_104/StatefulPartitionedCall/batch_normalization_104/StatefulPartitionedCall2`
.batch_normalization_98/StatefulPartitionedCall.batch_normalization_98/StatefulPartitionedCall2`
.batch_normalization_99/StatefulPartitionedCall.batch_normalization_99/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2Z
+depthwise_conv2d_42/StatefulPartitionedCall+depthwise_conv2d_42/StatefulPartitionedCall2Z
+depthwise_conv2d_43/StatefulPartitionedCall+depthwise_conv2d_43/StatefulPartitionedCall2Z
+depthwise_conv2d_44/StatefulPartitionedCall+depthwise_conv2d_44/StatefulPartitionedCall2Z
+separable_conv2d_42/StatefulPartitionedCall+separable_conv2d_42/StatefulPartitionedCall2Z
+separable_conv2d_43/StatefulPartitionedCall+separable_conv2d_43/StatefulPartitionedCall2Z
+separable_conv2d_44/StatefulPartitionedCall+separable_conv2d_44/StatefulPartitionedCall:Z V
0
_output_shapes
:??????????
"
_user_specified_name
input_15
?
d
+__inference_dropout_84_layer_call_fn_588139

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_84_layer_call_and_return_conditional_losses_5858492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????K2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
`
D__inference_flatten1_layer_call_and_return_conditional_losses_588868

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
I__inference_activation_87_layer_call_and_return_conditional_losses_588706

inputs
identityh
re_lu_87/ReluReluinputs*
T0*/
_output_shapes
:?????????K2
re_lu_87/Reluw
IdentityIdentityre_lu_87/Relu:activations:0*
T0*/
_output_shapes
:?????????K2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_100_layer_call_and_return_conditional_losses_588337

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_98_layer_call_fn_587554

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_98_layer_call_and_return_conditional_losses_5861172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_103_layer_call_and_return_conditional_losses_587916

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
e
F__inference_dropout_84_layer_call_and_return_conditional_losses_585849

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????K2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????K*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????K2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????K2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????K2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????K2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_102_layer_call_and_return_conditional_losses_588497

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????K:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????K2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????K: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?

?
O__inference_depthwise_conv2d_43_layer_call_and_return_conditional_losses_585036

inputs;
!depthwise_readvariableop_resource:
identity??depthwise/ReadVariableOp?
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
depthwise/Shape?
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rate?
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
	depthwisev
IdentityIdentitydepthwise:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityi
NoOpNoOp^depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
Q
5__inference_average_pooling2d_88_layer_call_fn_588112

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_88_layer_call_and_return_conditional_losses_5851582
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????K2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
4__inference_separable_conv2d_43_layer_call_fn_588268

inputs!
unknown:@#
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_separable_conv2d_43_layer_call_and_return_conditional_losses_5844552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

?
O__inference_depthwise_conv2d_44_layer_call_and_return_conditional_losses_587627

inputs;
!depthwise_readvariableop_resource:
identity??depthwise/ReadVariableOp?
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
depthwise/Shape?
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rate?
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
	depthwise?
IdentityIdentitydepthwise:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityi
NoOpNoOp^depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+???????????????????????????: 24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_102_layer_call_and_return_conditional_losses_588479

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_98_layer_call_and_return_conditional_losses_583745

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
d
+__inference_dropout_86_layer_call_fn_588166

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_86_layer_call_and_return_conditional_losses_5858722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????K2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
?
O__inference_separable_conv2d_44_layer_call_and_return_conditional_losses_584499

inputsC
(separable_conv2d_readvariableop_resource:?D
*separable_conv2d_readvariableop_1_resource:
identity??separable_conv2d/ReadVariableOp?!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02!
separable_conv2d/ReadVariableOp?
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype02#
!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"   ?         2
separable_conv2d/Shape?
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rate?
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
separable_conv2d/depthwise?
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
separable_conv2d?
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_98_layer_call_and_return_conditional_losses_587502

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
l
P__inference_average_pooling2d_85_layer_call_and_return_conditional_losses_585362

inputs
identity?
AvgPoolAvgPoolinputs*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
AvgPooll
IdentityIdentityAvgPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_103_layer_call_and_return_conditional_losses_587952

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
E__inference_flatten64_layer_call_and_return_conditional_losses_585399

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
O__inference_separable_conv2d_44_layer_call_and_return_conditional_losses_588301

inputsC
(separable_conv2d_readvariableop_resource:?D
*separable_conv2d_readvariableop_1_resource:
identity??separable_conv2d/ReadVariableOp?!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02!
separable_conv2d/ReadVariableOp?
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype02#
!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"   ?         2
separable_conv2d/Shape?
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rate?
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????K*
paddingSAME*
strides
2
separable_conv2d/depthwise?
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????K*
paddingVALID*
strides
2
separable_conv2d|
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*/
_output_shapes
:?????????K2

Identity?
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????K: : 2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
J
.__inference_activation_88_layer_call_fn_588052

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_88_layer_call_and_return_conditional_losses_5851382
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
l
P__inference_average_pooling2d_88_layer_call_and_return_conditional_losses_588097

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
AvgPool?
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
e
I__inference_activation_84_layer_call_and_return_conditional_losses_588027

inputs
identityi
re_lu_84/ReluReluinputs*
T0*0
_output_shapes
:??????????2
re_lu_84/Relux
IdentityIdentityre_lu_84/Relu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_100_layer_call_fn_588404

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_100_layer_call_and_return_conditional_losses_5845492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_100_layer_call_and_return_conditional_losses_584593

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_89_layer_call_and_return_conditional_losses_585369

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
4__inference_separable_conv2d_42_layer_call_fn_588226

inputs!
unknown: #
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_separable_conv2d_42_layer_call_and_return_conditional_losses_5844112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_104_layer_call_and_return_conditional_losses_584845

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_flatten128_layer_call_and_return_conditional_losses_588890

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_101_layer_call_and_return_conditional_losses_585096

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
4__inference_separable_conv2d_44_layer_call_fn_588310

inputs"
unknown:?#
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_separable_conv2d_44_layer_call_and_return_conditional_losses_5844992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_87_layer_call_and_return_conditional_losses_588813

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
O__inference_depthwise_conv2d_43_layer_call_and_return_conditional_losses_587604

inputs;
!depthwise_readvariableop_resource:
identity??depthwise/ReadVariableOp?
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
depthwise/Shape?
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rate?
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
	depthwisev
IdentityIdentitydepthwise:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityi
NoOpNoOp^depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
4__inference_depthwise_conv2d_43_layer_call_fn_587611

inputs!
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_depthwise_conv2d_43_layer_call_and_return_conditional_losses_5838962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+???????????????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_104_layer_call_fn_588691

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_104_layer_call_and_return_conditional_losses_5857912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????K2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????K: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_101_layer_call_fn_587898

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_101_layer_call_and_return_conditional_losses_5860082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_102_layer_call_fn_588567

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_102_layer_call_and_return_conditional_losses_5857472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????K2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????K: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
J
.__inference_activation_85_layer_call_fn_588701

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_85_layer_call_and_return_conditional_losses_5853442
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????K2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
D
(__inference_softmax_layer_call_fn_588939

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_softmax_layer_call_and_return_conditional_losses_5854402
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
O__inference_separable_conv2d_43_layer_call_and_return_conditional_losses_585221

inputsB
(separable_conv2d_readvariableop_resource:@D
*separable_conv2d_readvariableop_1_resource:
identity??separable_conv2d/ReadVariableOp?!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02!
separable_conv2d/ReadVariableOp?
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype02#
!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"   @         2
separable_conv2d/Shape?
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rate?
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????K*
paddingSAME*
strides
2
separable_conv2d/depthwise?
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????K*
paddingVALID*
strides
2
separable_conv2d|
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*/
_output_shapes
:?????????K2

Identity?
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????K: : 2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_102_layer_call_and_return_conditional_losses_584675

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_102_layer_call_and_return_conditional_losses_584719

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
l
P__inference_average_pooling2d_84_layer_call_and_return_conditional_losses_588057

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
AvgPool?
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_102_layer_call_fn_588554

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_102_layer_call_and_return_conditional_losses_5852882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????K2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????K: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_98_layer_call_fn_587528

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_98_layer_call_and_return_conditional_losses_5837892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
e
F__inference_dropout_88_layer_call_and_return_conditional_losses_585895

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????K2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????K*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????K2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????K2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????K2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????K2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
?
4__inference_separable_conv2d_44_layer_call_fn_588319

inputs"
unknown:?#
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_separable_conv2d_44_layer_call_and_return_conditional_losses_5852042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????K2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????K: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
e
F__inference_dropout_85_layer_call_and_return_conditional_losses_588798

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
I__inference_activation_85_layer_call_and_return_conditional_losses_588696

inputs
identityh
re_lu_85/ReluReluinputs*
T0*/
_output_shapes
:?????????K2
re_lu_85/Reluw
IdentityIdentityre_lu_85/Relu:activations:0*
T0*/
_output_shapes
:?????????K2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_99_layer_call_fn_587735

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_99_layer_call_and_return_conditional_losses_5839732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
4__inference_depthwise_conv2d_42_layer_call_fn_587586

inputs!
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_depthwise_conv2d_42_layer_call_and_return_conditional_losses_5850482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_14_layer_call_and_return_conditional_losses_584985

inputs8
conv2d_readvariableop_resource:d
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:d*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2Ds
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityf
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
J
.__inference_activation_89_layer_call_fn_588721

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_89_layer_call_and_return_conditional_losses_5853302
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????K2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
ʺ
?
D__inference_model_14_layer_call_and_return_conditional_losses_586757
input_15*
conv2d_14_586638:d+
batch_normalization_98_586641:+
batch_normalization_98_586643:+
batch_normalization_98_586645:+
batch_normalization_98_586647:4
depthwise_conv2d_44_586650:4
depthwise_conv2d_43_586653:4
depthwise_conv2d_42_586656:,
batch_normalization_103_586659:,
batch_normalization_103_586661:,
batch_normalization_103_586663:,
batch_normalization_103_586665:,
batch_normalization_101_586668:,
batch_normalization_101_586670:,
batch_normalization_101_586672:,
batch_normalization_101_586674:+
batch_normalization_99_586677:+
batch_normalization_99_586679:+
batch_normalization_99_586681:+
batch_normalization_99_586683:5
separable_conv2d_44_586695:?4
separable_conv2d_44_586697:4
separable_conv2d_43_586700:@4
separable_conv2d_43_586702:4
separable_conv2d_42_586705: 4
separable_conv2d_42_586707:,
batch_normalization_104_586710:,
batch_normalization_104_586712:,
batch_normalization_104_586714:,
batch_normalization_104_586716:,
batch_normalization_102_586719:,
batch_normalization_102_586721:,
batch_normalization_102_586723:,
batch_normalization_102_586725:,
batch_normalization_100_586728:,
batch_normalization_100_586730:,
batch_normalization_100_586732:,
batch_normalization_100_586734: 
dense2_586750:	?
dense2_586752:
identity??/batch_normalization_100/StatefulPartitionedCall?/batch_normalization_101/StatefulPartitionedCall?/batch_normalization_102/StatefulPartitionedCall?/batch_normalization_103/StatefulPartitionedCall?/batch_normalization_104/StatefulPartitionedCall?.batch_normalization_98/StatefulPartitionedCall?.batch_normalization_99/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?dense2/StatefulPartitionedCall?+depthwise_conv2d_42/StatefulPartitionedCall?+depthwise_conv2d_43/StatefulPartitionedCall?+depthwise_conv2d_44/StatefulPartitionedCall?"dropout_84/StatefulPartitionedCall?"dropout_85/StatefulPartitionedCall?"dropout_86/StatefulPartitionedCall?"dropout_87/StatefulPartitionedCall?"dropout_88/StatefulPartitionedCall?"dropout_89/StatefulPartitionedCall?+separable_conv2d_42/StatefulPartitionedCall?+separable_conv2d_43/StatefulPartitionedCall?+separable_conv2d_44/StatefulPartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCallinput_15conv2d_14_586638*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_5849852#
!conv2d_14/StatefulPartitionedCall?
.batch_normalization_98/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_98_586641batch_normalization_98_586643batch_normalization_98_586645batch_normalization_98_586647*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_98_layer_call_and_return_conditional_losses_58611720
.batch_normalization_98/StatefulPartitionedCall?
+depthwise_conv2d_44/StatefulPartitionedCallStatefulPartitionedCallinput_15depthwise_conv2d_44_586650*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_depthwise_conv2d_44_layer_call_and_return_conditional_losses_5850242-
+depthwise_conv2d_44/StatefulPartitionedCall?
+depthwise_conv2d_43/StatefulPartitionedCallStatefulPartitionedCallinput_15depthwise_conv2d_43_586653*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_depthwise_conv2d_43_layer_call_and_return_conditional_losses_5850362-
+depthwise_conv2d_43/StatefulPartitionedCall?
+depthwise_conv2d_42/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_98/StatefulPartitionedCall:output:0depthwise_conv2d_42_586656*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_depthwise_conv2d_42_layer_call_and_return_conditional_losses_5850482-
+depthwise_conv2d_42/StatefulPartitionedCall?
/batch_normalization_103/StatefulPartitionedCallStatefulPartitionedCall4depthwise_conv2d_44/StatefulPartitionedCall:output:0batch_normalization_103_586659batch_normalization_103_586661batch_normalization_103_586663batch_normalization_103_586665*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_103_layer_call_and_return_conditional_losses_58605221
/batch_normalization_103/StatefulPartitionedCall?
/batch_normalization_101/StatefulPartitionedCallStatefulPartitionedCall4depthwise_conv2d_43/StatefulPartitionedCall:output:0batch_normalization_101_586668batch_normalization_101_586670batch_normalization_101_586672batch_normalization_101_586674*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_101_layer_call_and_return_conditional_losses_58600821
/batch_normalization_101/StatefulPartitionedCall?
.batch_normalization_99/StatefulPartitionedCallStatefulPartitionedCall4depthwise_conv2d_42/StatefulPartitionedCall:output:0batch_normalization_99_586677batch_normalization_99_586679batch_normalization_99_586681batch_normalization_99_586683*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_99_layer_call_and_return_conditional_losses_58596420
.batch_normalization_99/StatefulPartitionedCall?
activation_88/PartitionedCallPartitionedCall8batch_normalization_103/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_88_layer_call_and_return_conditional_losses_5851382
activation_88/PartitionedCall?
activation_86/PartitionedCallPartitionedCall8batch_normalization_101/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_86_layer_call_and_return_conditional_losses_5851452
activation_86/PartitionedCall?
activation_84/PartitionedCallPartitionedCall7batch_normalization_99/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_84_layer_call_and_return_conditional_losses_5851522
activation_84/PartitionedCall?
$average_pooling2d_88/PartitionedCallPartitionedCall&activation_88/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_88_layer_call_and_return_conditional_losses_5851582&
$average_pooling2d_88/PartitionedCall?
$average_pooling2d_86/PartitionedCallPartitionedCall&activation_86/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_86_layer_call_and_return_conditional_losses_5851642&
$average_pooling2d_86/PartitionedCall?
$average_pooling2d_84/PartitionedCallPartitionedCall&activation_84/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_84_layer_call_and_return_conditional_losses_5851702&
$average_pooling2d_84/PartitionedCall?
"dropout_88/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_88/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_88_layer_call_and_return_conditional_losses_5858952$
"dropout_88/StatefulPartitionedCall?
"dropout_86/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_86/PartitionedCall:output:0#^dropout_88/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_86_layer_call_and_return_conditional_losses_5858722$
"dropout_86/StatefulPartitionedCall?
"dropout_84/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_84/PartitionedCall:output:0#^dropout_86/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_84_layer_call_and_return_conditional_losses_5858492$
"dropout_84/StatefulPartitionedCall?
+separable_conv2d_44/StatefulPartitionedCallStatefulPartitionedCall+dropout_88/StatefulPartitionedCall:output:0separable_conv2d_44_586695separable_conv2d_44_586697*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_separable_conv2d_44_layer_call_and_return_conditional_losses_5852042-
+separable_conv2d_44/StatefulPartitionedCall?
+separable_conv2d_43/StatefulPartitionedCallStatefulPartitionedCall+dropout_86/StatefulPartitionedCall:output:0separable_conv2d_43_586700separable_conv2d_43_586702*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_separable_conv2d_43_layer_call_and_return_conditional_losses_5852212-
+separable_conv2d_43/StatefulPartitionedCall?
+separable_conv2d_42/StatefulPartitionedCallStatefulPartitionedCall+dropout_84/StatefulPartitionedCall:output:0separable_conv2d_42_586705separable_conv2d_42_586707*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_separable_conv2d_42_layer_call_and_return_conditional_losses_5852382-
+separable_conv2d_42/StatefulPartitionedCall?
/batch_normalization_104/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_44/StatefulPartitionedCall:output:0batch_normalization_104_586710batch_normalization_104_586712batch_normalization_104_586714batch_normalization_104_586716*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_104_layer_call_and_return_conditional_losses_58579121
/batch_normalization_104/StatefulPartitionedCall?
/batch_normalization_102/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_43/StatefulPartitionedCall:output:0batch_normalization_102_586719batch_normalization_102_586721batch_normalization_102_586723batch_normalization_102_586725*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_102_layer_call_and_return_conditional_losses_58574721
/batch_normalization_102/StatefulPartitionedCall?
/batch_normalization_100/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_42/StatefulPartitionedCall:output:0batch_normalization_100_586728batch_normalization_100_586730batch_normalization_100_586732batch_normalization_100_586734*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_100_layer_call_and_return_conditional_losses_58570321
/batch_normalization_100/StatefulPartitionedCall?
activation_89/PartitionedCallPartitionedCall8batch_normalization_104/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_89_layer_call_and_return_conditional_losses_5853302
activation_89/PartitionedCall?
activation_87/PartitionedCallPartitionedCall8batch_normalization_102/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_87_layer_call_and_return_conditional_losses_5853372
activation_87/PartitionedCall?
activation_85/PartitionedCallPartitionedCall8batch_normalization_100/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_85_layer_call_and_return_conditional_losses_5853442
activation_85/PartitionedCall?
$average_pooling2d_89/PartitionedCallPartitionedCall&activation_89/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_89_layer_call_and_return_conditional_losses_5853502&
$average_pooling2d_89/PartitionedCall?
$average_pooling2d_87/PartitionedCallPartitionedCall&activation_87/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_87_layer_call_and_return_conditional_losses_5853562&
$average_pooling2d_87/PartitionedCall?
$average_pooling2d_85/PartitionedCallPartitionedCall&activation_85/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_85_layer_call_and_return_conditional_losses_5853622&
$average_pooling2d_85/PartitionedCall?
"dropout_89/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_89/PartitionedCall:output:0#^dropout_84/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_89_layer_call_and_return_conditional_losses_5856342$
"dropout_89/StatefulPartitionedCall?
"dropout_87/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_87/PartitionedCall:output:0#^dropout_89/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_87_layer_call_and_return_conditional_losses_5856112$
"dropout_87/StatefulPartitionedCall?
"dropout_85/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_85/PartitionedCall:output:0#^dropout_87/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_85_layer_call_and_return_conditional_losses_5855882$
"dropout_85/StatefulPartitionedCall?
flatten1/PartitionedCallPartitionedCall+dropout_85/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten1_layer_call_and_return_conditional_losses_5853912
flatten1/PartitionedCall?
flatten64/PartitionedCallPartitionedCall+dropout_87/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten64_layer_call_and_return_conditional_losses_5853992
flatten64/PartitionedCall?
flatten128/PartitionedCallPartitionedCall+dropout_89/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten128_layer_call_and_return_conditional_losses_5854072
flatten128/PartitionedCall?
concatenate_14/PartitionedCallPartitionedCall!flatten1/PartitionedCall:output:0"flatten64/PartitionedCall:output:0#flatten128/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_concatenate_14_layer_call_and_return_conditional_losses_5854172 
concatenate_14/PartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall'concatenate_14/PartitionedCall:output:0dense2_586750dense2_586752*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense2_layer_call_and_return_conditional_losses_5854292 
dense2/StatefulPartitionedCall?
softmax/PartitionedCallPartitionedCall'dense2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_softmax_layer_call_and_return_conditional_losses_5854402
softmax/PartitionedCall{
IdentityIdentity softmax/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp0^batch_normalization_100/StatefulPartitionedCall0^batch_normalization_101/StatefulPartitionedCall0^batch_normalization_102/StatefulPartitionedCall0^batch_normalization_103/StatefulPartitionedCall0^batch_normalization_104/StatefulPartitionedCall/^batch_normalization_98/StatefulPartitionedCall/^batch_normalization_99/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall^dense2/StatefulPartitionedCall,^depthwise_conv2d_42/StatefulPartitionedCall,^depthwise_conv2d_43/StatefulPartitionedCall,^depthwise_conv2d_44/StatefulPartitionedCall#^dropout_84/StatefulPartitionedCall#^dropout_85/StatefulPartitionedCall#^dropout_86/StatefulPartitionedCall#^dropout_87/StatefulPartitionedCall#^dropout_88/StatefulPartitionedCall#^dropout_89/StatefulPartitionedCall,^separable_conv2d_42/StatefulPartitionedCall,^separable_conv2d_43/StatefulPartitionedCall,^separable_conv2d_44/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesn
l:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_100/StatefulPartitionedCall/batch_normalization_100/StatefulPartitionedCall2b
/batch_normalization_101/StatefulPartitionedCall/batch_normalization_101/StatefulPartitionedCall2b
/batch_normalization_102/StatefulPartitionedCall/batch_normalization_102/StatefulPartitionedCall2b
/batch_normalization_103/StatefulPartitionedCall/batch_normalization_103/StatefulPartitionedCall2b
/batch_normalization_104/StatefulPartitionedCall/batch_normalization_104/StatefulPartitionedCall2`
.batch_normalization_98/StatefulPartitionedCall.batch_normalization_98/StatefulPartitionedCall2`
.batch_normalization_99/StatefulPartitionedCall.batch_normalization_99/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2Z
+depthwise_conv2d_42/StatefulPartitionedCall+depthwise_conv2d_42/StatefulPartitionedCall2Z
+depthwise_conv2d_43/StatefulPartitionedCall+depthwise_conv2d_43/StatefulPartitionedCall2Z
+depthwise_conv2d_44/StatefulPartitionedCall+depthwise_conv2d_44/StatefulPartitionedCall2H
"dropout_84/StatefulPartitionedCall"dropout_84/StatefulPartitionedCall2H
"dropout_85/StatefulPartitionedCall"dropout_85/StatefulPartitionedCall2H
"dropout_86/StatefulPartitionedCall"dropout_86/StatefulPartitionedCall2H
"dropout_87/StatefulPartitionedCall"dropout_87/StatefulPartitionedCall2H
"dropout_88/StatefulPartitionedCall"dropout_88/StatefulPartitionedCall2H
"dropout_89/StatefulPartitionedCall"dropout_89/StatefulPartitionedCall2Z
+separable_conv2d_42/StatefulPartitionedCall+separable_conv2d_42/StatefulPartitionedCall2Z
+separable_conv2d_43/StatefulPartitionedCall+separable_conv2d_43/StatefulPartitionedCall2Z
+separable_conv2d_44/StatefulPartitionedCall+separable_conv2d_44/StatefulPartitionedCall:Z V
0
_output_shapes
:??????????
"
_user_specified_name
input_15
?
e
F__inference_dropout_89_layer_call_and_return_conditional_losses_588852

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
l
P__inference_average_pooling2d_89_layer_call_and_return_conditional_losses_584958

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
AvgPool?
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_103_layer_call_and_return_conditional_losses_587970

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
l
P__inference_average_pooling2d_89_layer_call_and_return_conditional_losses_588771

inputs
identity?
AvgPoolAvgPoolinputs*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
AvgPooll
IdentityIdentityAvgPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
l
P__inference_average_pooling2d_86_layer_call_and_return_conditional_losses_585164

inputs
identity?
AvgPoolAvgPoolinputs*
T0*/
_output_shapes
:?????????K*
ksize
*
paddingVALID*
strides
2	
AvgPooll
IdentityIdentityAvgPool:output:0*
T0*/
_output_shapes
:?????????K2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
J__inference_concatenate_14_layer_call_and_return_conditional_losses_585417

inputs
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????:??????????:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
4__inference_depthwise_conv2d_43_layer_call_fn_587618

inputs!
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_depthwise_conv2d_43_layer_call_and_return_conditional_losses_5850362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
l
P__inference_average_pooling2d_85_layer_call_and_return_conditional_losses_588726

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
AvgPool?
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
l
P__inference_average_pooling2d_84_layer_call_and_return_conditional_losses_584338

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
AvgPool?
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
O__inference_depthwise_conv2d_43_layer_call_and_return_conditional_losses_583896

inputs;
!depthwise_readvariableop_resource:
identity??depthwise/ReadVariableOp?
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
depthwise/Shape?
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rate?
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
	depthwise?
IdentityIdentitydepthwise:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityi
NoOpNoOp^depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+???????????????????????????: 24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
ݰ
?
D__inference_model_14_layer_call_and_return_conditional_losses_585443

inputs*
conv2d_14_584986:d+
batch_normalization_98_585007:+
batch_normalization_98_585009:+
batch_normalization_98_585011:+
batch_normalization_98_585013:4
depthwise_conv2d_44_585025:4
depthwise_conv2d_43_585037:4
depthwise_conv2d_42_585049:,
batch_normalization_103_585070:,
batch_normalization_103_585072:,
batch_normalization_103_585074:,
batch_normalization_103_585076:,
batch_normalization_101_585097:,
batch_normalization_101_585099:,
batch_normalization_101_585101:,
batch_normalization_101_585103:+
batch_normalization_99_585124:+
batch_normalization_99_585126:+
batch_normalization_99_585128:+
batch_normalization_99_585130:5
separable_conv2d_44_585205:?4
separable_conv2d_44_585207:4
separable_conv2d_43_585222:@4
separable_conv2d_43_585224:4
separable_conv2d_42_585239: 4
separable_conv2d_42_585241:,
batch_normalization_104_585262:,
batch_normalization_104_585264:,
batch_normalization_104_585266:,
batch_normalization_104_585268:,
batch_normalization_102_585289:,
batch_normalization_102_585291:,
batch_normalization_102_585293:,
batch_normalization_102_585295:,
batch_normalization_100_585316:,
batch_normalization_100_585318:,
batch_normalization_100_585320:,
batch_normalization_100_585322: 
dense2_585430:	?
dense2_585432:
identity??/batch_normalization_100/StatefulPartitionedCall?/batch_normalization_101/StatefulPartitionedCall?/batch_normalization_102/StatefulPartitionedCall?/batch_normalization_103/StatefulPartitionedCall?/batch_normalization_104/StatefulPartitionedCall?.batch_normalization_98/StatefulPartitionedCall?.batch_normalization_99/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?dense2/StatefulPartitionedCall?+depthwise_conv2d_42/StatefulPartitionedCall?+depthwise_conv2d_43/StatefulPartitionedCall?+depthwise_conv2d_44/StatefulPartitionedCall?+separable_conv2d_42/StatefulPartitionedCall?+separable_conv2d_43/StatefulPartitionedCall?+separable_conv2d_44/StatefulPartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_14_584986*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_5849852#
!conv2d_14/StatefulPartitionedCall?
.batch_normalization_98/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_98_585007batch_normalization_98_585009batch_normalization_98_585011batch_normalization_98_585013*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_98_layer_call_and_return_conditional_losses_58500620
.batch_normalization_98/StatefulPartitionedCall?
+depthwise_conv2d_44/StatefulPartitionedCallStatefulPartitionedCallinputsdepthwise_conv2d_44_585025*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_depthwise_conv2d_44_layer_call_and_return_conditional_losses_5850242-
+depthwise_conv2d_44/StatefulPartitionedCall?
+depthwise_conv2d_43/StatefulPartitionedCallStatefulPartitionedCallinputsdepthwise_conv2d_43_585037*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_depthwise_conv2d_43_layer_call_and_return_conditional_losses_5850362-
+depthwise_conv2d_43/StatefulPartitionedCall?
+depthwise_conv2d_42/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_98/StatefulPartitionedCall:output:0depthwise_conv2d_42_585049*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_depthwise_conv2d_42_layer_call_and_return_conditional_losses_5850482-
+depthwise_conv2d_42/StatefulPartitionedCall?
/batch_normalization_103/StatefulPartitionedCallStatefulPartitionedCall4depthwise_conv2d_44/StatefulPartitionedCall:output:0batch_normalization_103_585070batch_normalization_103_585072batch_normalization_103_585074batch_normalization_103_585076*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_103_layer_call_and_return_conditional_losses_58506921
/batch_normalization_103/StatefulPartitionedCall?
/batch_normalization_101/StatefulPartitionedCallStatefulPartitionedCall4depthwise_conv2d_43/StatefulPartitionedCall:output:0batch_normalization_101_585097batch_normalization_101_585099batch_normalization_101_585101batch_normalization_101_585103*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_101_layer_call_and_return_conditional_losses_58509621
/batch_normalization_101/StatefulPartitionedCall?
.batch_normalization_99/StatefulPartitionedCallStatefulPartitionedCall4depthwise_conv2d_42/StatefulPartitionedCall:output:0batch_normalization_99_585124batch_normalization_99_585126batch_normalization_99_585128batch_normalization_99_585130*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_99_layer_call_and_return_conditional_losses_58512320
.batch_normalization_99/StatefulPartitionedCall?
activation_88/PartitionedCallPartitionedCall8batch_normalization_103/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_88_layer_call_and_return_conditional_losses_5851382
activation_88/PartitionedCall?
activation_86/PartitionedCallPartitionedCall8batch_normalization_101/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_86_layer_call_and_return_conditional_losses_5851452
activation_86/PartitionedCall?
activation_84/PartitionedCallPartitionedCall7batch_normalization_99/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_84_layer_call_and_return_conditional_losses_5851522
activation_84/PartitionedCall?
$average_pooling2d_88/PartitionedCallPartitionedCall&activation_88/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_88_layer_call_and_return_conditional_losses_5851582&
$average_pooling2d_88/PartitionedCall?
$average_pooling2d_86/PartitionedCallPartitionedCall&activation_86/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_86_layer_call_and_return_conditional_losses_5851642&
$average_pooling2d_86/PartitionedCall?
$average_pooling2d_84/PartitionedCallPartitionedCall&activation_84/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_84_layer_call_and_return_conditional_losses_5851702&
$average_pooling2d_84/PartitionedCall?
dropout_88/PartitionedCallPartitionedCall-average_pooling2d_88/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_88_layer_call_and_return_conditional_losses_5851772
dropout_88/PartitionedCall?
dropout_86/PartitionedCallPartitionedCall-average_pooling2d_86/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_86_layer_call_and_return_conditional_losses_5851842
dropout_86/PartitionedCall?
dropout_84/PartitionedCallPartitionedCall-average_pooling2d_84/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_84_layer_call_and_return_conditional_losses_5851912
dropout_84/PartitionedCall?
+separable_conv2d_44/StatefulPartitionedCallStatefulPartitionedCall#dropout_88/PartitionedCall:output:0separable_conv2d_44_585205separable_conv2d_44_585207*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_separable_conv2d_44_layer_call_and_return_conditional_losses_5852042-
+separable_conv2d_44/StatefulPartitionedCall?
+separable_conv2d_43/StatefulPartitionedCallStatefulPartitionedCall#dropout_86/PartitionedCall:output:0separable_conv2d_43_585222separable_conv2d_43_585224*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_separable_conv2d_43_layer_call_and_return_conditional_losses_5852212-
+separable_conv2d_43/StatefulPartitionedCall?
+separable_conv2d_42/StatefulPartitionedCallStatefulPartitionedCall#dropout_84/PartitionedCall:output:0separable_conv2d_42_585239separable_conv2d_42_585241*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_separable_conv2d_42_layer_call_and_return_conditional_losses_5852382-
+separable_conv2d_42/StatefulPartitionedCall?
/batch_normalization_104/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_44/StatefulPartitionedCall:output:0batch_normalization_104_585262batch_normalization_104_585264batch_normalization_104_585266batch_normalization_104_585268*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_104_layer_call_and_return_conditional_losses_58526121
/batch_normalization_104/StatefulPartitionedCall?
/batch_normalization_102/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_43/StatefulPartitionedCall:output:0batch_normalization_102_585289batch_normalization_102_585291batch_normalization_102_585293batch_normalization_102_585295*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_102_layer_call_and_return_conditional_losses_58528821
/batch_normalization_102/StatefulPartitionedCall?
/batch_normalization_100/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_42/StatefulPartitionedCall:output:0batch_normalization_100_585316batch_normalization_100_585318batch_normalization_100_585320batch_normalization_100_585322*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_100_layer_call_and_return_conditional_losses_58531521
/batch_normalization_100/StatefulPartitionedCall?
activation_89/PartitionedCallPartitionedCall8batch_normalization_104/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_89_layer_call_and_return_conditional_losses_5853302
activation_89/PartitionedCall?
activation_87/PartitionedCallPartitionedCall8batch_normalization_102/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_87_layer_call_and_return_conditional_losses_5853372
activation_87/PartitionedCall?
activation_85/PartitionedCallPartitionedCall8batch_normalization_100/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_85_layer_call_and_return_conditional_losses_5853442
activation_85/PartitionedCall?
$average_pooling2d_89/PartitionedCallPartitionedCall&activation_89/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_89_layer_call_and_return_conditional_losses_5853502&
$average_pooling2d_89/PartitionedCall?
$average_pooling2d_87/PartitionedCallPartitionedCall&activation_87/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_87_layer_call_and_return_conditional_losses_5853562&
$average_pooling2d_87/PartitionedCall?
$average_pooling2d_85/PartitionedCallPartitionedCall&activation_85/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_85_layer_call_and_return_conditional_losses_5853622&
$average_pooling2d_85/PartitionedCall?
dropout_89/PartitionedCallPartitionedCall-average_pooling2d_89/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_89_layer_call_and_return_conditional_losses_5853692
dropout_89/PartitionedCall?
dropout_87/PartitionedCallPartitionedCall-average_pooling2d_87/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_87_layer_call_and_return_conditional_losses_5853762
dropout_87/PartitionedCall?
dropout_85/PartitionedCallPartitionedCall-average_pooling2d_85/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_85_layer_call_and_return_conditional_losses_5853832
dropout_85/PartitionedCall?
flatten1/PartitionedCallPartitionedCall#dropout_85/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten1_layer_call_and_return_conditional_losses_5853912
flatten1/PartitionedCall?
flatten64/PartitionedCallPartitionedCall#dropout_87/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten64_layer_call_and_return_conditional_losses_5853992
flatten64/PartitionedCall?
flatten128/PartitionedCallPartitionedCall#dropout_89/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten128_layer_call_and_return_conditional_losses_5854072
flatten128/PartitionedCall?
concatenate_14/PartitionedCallPartitionedCall!flatten1/PartitionedCall:output:0"flatten64/PartitionedCall:output:0#flatten128/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_concatenate_14_layer_call_and_return_conditional_losses_5854172 
concatenate_14/PartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall'concatenate_14/PartitionedCall:output:0dense2_585430dense2_585432*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense2_layer_call_and_return_conditional_losses_5854292 
dense2/StatefulPartitionedCall?
softmax/PartitionedCallPartitionedCall'dense2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_softmax_layer_call_and_return_conditional_losses_5854402
softmax/PartitionedCall{
IdentityIdentity softmax/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp0^batch_normalization_100/StatefulPartitionedCall0^batch_normalization_101/StatefulPartitionedCall0^batch_normalization_102/StatefulPartitionedCall0^batch_normalization_103/StatefulPartitionedCall0^batch_normalization_104/StatefulPartitionedCall/^batch_normalization_98/StatefulPartitionedCall/^batch_normalization_99/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall^dense2/StatefulPartitionedCall,^depthwise_conv2d_42/StatefulPartitionedCall,^depthwise_conv2d_43/StatefulPartitionedCall,^depthwise_conv2d_44/StatefulPartitionedCall,^separable_conv2d_42/StatefulPartitionedCall,^separable_conv2d_43/StatefulPartitionedCall,^separable_conv2d_44/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesn
l:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_100/StatefulPartitionedCall/batch_normalization_100/StatefulPartitionedCall2b
/batch_normalization_101/StatefulPartitionedCall/batch_normalization_101/StatefulPartitionedCall2b
/batch_normalization_102/StatefulPartitionedCall/batch_normalization_102/StatefulPartitionedCall2b
/batch_normalization_103/StatefulPartitionedCall/batch_normalization_103/StatefulPartitionedCall2b
/batch_normalization_104/StatefulPartitionedCall/batch_normalization_104/StatefulPartitionedCall2`
.batch_normalization_98/StatefulPartitionedCall.batch_normalization_98/StatefulPartitionedCall2`
.batch_normalization_99/StatefulPartitionedCall.batch_normalization_99/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2Z
+depthwise_conv2d_42/StatefulPartitionedCall+depthwise_conv2d_42/StatefulPartitionedCall2Z
+depthwise_conv2d_43/StatefulPartitionedCall+depthwise_conv2d_43/StatefulPartitionedCall2Z
+depthwise_conv2d_44/StatefulPartitionedCall+depthwise_conv2d_44/StatefulPartitionedCall2Z
+separable_conv2d_42/StatefulPartitionedCall+separable_conv2d_42/StatefulPartitionedCall2Z
+separable_conv2d_43/StatefulPartitionedCall+separable_conv2d_43/StatefulPartitionedCall2Z
+separable_conv2d_44/StatefulPartitionedCall+separable_conv2d_44/StatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_104_layer_call_and_return_conditional_losses_584801

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_99_layer_call_and_return_conditional_losses_587686

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
Q
5__inference_average_pooling2d_89_layer_call_fn_588776

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_89_layer_call_and_return_conditional_losses_5849582
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_98_layer_call_fn_587515

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_98_layer_call_and_return_conditional_losses_5837452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
e
F__inference_dropout_86_layer_call_and_return_conditional_losses_588156

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????K2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????K*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????K2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????K2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????K2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????K2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
d
+__inference_dropout_87_layer_call_fn_588835

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_87_layer_call_and_return_conditional_losses_5856112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
l
P__inference_average_pooling2d_88_layer_call_and_return_conditional_losses_588102

inputs
identity?
AvgPoolAvgPoolinputs*
T0*/
_output_shapes
:?????????K*
ksize
*
paddingVALID*
strides
2	
AvgPooll
IdentityIdentityAvgPool:output:0*
T0*/
_output_shapes
:?????????K2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
O__inference_depthwise_conv2d_43_layer_call_and_return_conditional_losses_587595

inputs;
!depthwise_readvariableop_resource:
identity??depthwise/ReadVariableOp?
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
depthwise/Shape?
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rate?
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
	depthwise?
IdentityIdentitydepthwise:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityi
NoOpNoOp^depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+???????????????????????????: 24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_89_layer_call_and_return_conditional_losses_588840

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
O__inference_separable_conv2d_43_layer_call_and_return_conditional_losses_584455

inputsB
(separable_conv2d_readvariableop_resource:@D
*separable_conv2d_readvariableop_1_resource:
identity??separable_conv2d/ReadVariableOp?!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02!
separable_conv2d/ReadVariableOp?
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype02#
!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"   @         2
separable_conv2d/Shape?
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rate?
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
separable_conv2d/depthwise?
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
separable_conv2d?
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_separable_conv2d_42_layer_call_and_return_conditional_losses_584411

inputsB
(separable_conv2d_readvariableop_resource: D
*separable_conv2d_readvariableop_1_resource:
identity??separable_conv2d/ReadVariableOp?!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
separable_conv2d/ReadVariableOp?
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype02#
!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2
separable_conv2d/Shape?
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rate?
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
separable_conv2d/depthwise?
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
separable_conv2d?
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_102_layer_call_and_return_conditional_losses_588515

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????K:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????K2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????K: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
d
+__inference_dropout_88_layer_call_fn_588193

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_88_layer_call_and_return_conditional_losses_5858952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????K2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_98_layer_call_and_return_conditional_losses_586117

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
Q
5__inference_average_pooling2d_88_layer_call_fn_588107

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_88_layer_call_and_return_conditional_losses_5843822
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?	
)__inference_model_14_layer_call_fn_586513
input_15!
unknown:d
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:#
	unknown_4:#
	unknown_5:#
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:%

unknown_19:?$

unknown_20:$

unknown_21:@$

unknown_22:$

unknown_23: $

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:	?

unknown_38:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_15unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 #$'(*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_14_layer_call_and_return_conditional_losses_5863452
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesn
l:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:??????????
"
_user_specified_name
input_15
?
?
O__inference_separable_conv2d_42_layer_call_and_return_conditional_losses_588205

inputsB
(separable_conv2d_readvariableop_resource: D
*separable_conv2d_readvariableop_1_resource:
identity??separable_conv2d/ReadVariableOp?!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
separable_conv2d/ReadVariableOp?
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype02#
!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2
separable_conv2d/Shape?
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rate?
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
separable_conv2d/depthwise?
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
separable_conv2d?
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
e
F__inference_dropout_84_layer_call_and_return_conditional_losses_588129

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????K2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????K*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????K2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????K2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????K2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????K2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
d
F__inference_dropout_85_layer_call_and_return_conditional_losses_585383

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_104_layer_call_and_return_conditional_losses_585791

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????K:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????K2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????K: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
լ
?,
!__inference__wrapped_model_583723
input_15K
1model_14_conv2d_14_conv2d_readvariableop_resource:dE
7model_14_batch_normalization_98_readvariableop_resource:G
9model_14_batch_normalization_98_readvariableop_1_resource:V
Hmodel_14_batch_normalization_98_fusedbatchnormv3_readvariableop_resource:X
Jmodel_14_batch_normalization_98_fusedbatchnormv3_readvariableop_1_resource:X
>model_14_depthwise_conv2d_44_depthwise_readvariableop_resource:X
>model_14_depthwise_conv2d_43_depthwise_readvariableop_resource:X
>model_14_depthwise_conv2d_42_depthwise_readvariableop_resource:F
8model_14_batch_normalization_103_readvariableop_resource:H
:model_14_batch_normalization_103_readvariableop_1_resource:W
Imodel_14_batch_normalization_103_fusedbatchnormv3_readvariableop_resource:Y
Kmodel_14_batch_normalization_103_fusedbatchnormv3_readvariableop_1_resource:F
8model_14_batch_normalization_101_readvariableop_resource:H
:model_14_batch_normalization_101_readvariableop_1_resource:W
Imodel_14_batch_normalization_101_fusedbatchnormv3_readvariableop_resource:Y
Kmodel_14_batch_normalization_101_fusedbatchnormv3_readvariableop_1_resource:E
7model_14_batch_normalization_99_readvariableop_resource:G
9model_14_batch_normalization_99_readvariableop_1_resource:V
Hmodel_14_batch_normalization_99_fusedbatchnormv3_readvariableop_resource:X
Jmodel_14_batch_normalization_99_fusedbatchnormv3_readvariableop_1_resource:`
Emodel_14_separable_conv2d_44_separable_conv2d_readvariableop_resource:?a
Gmodel_14_separable_conv2d_44_separable_conv2d_readvariableop_1_resource:_
Emodel_14_separable_conv2d_43_separable_conv2d_readvariableop_resource:@a
Gmodel_14_separable_conv2d_43_separable_conv2d_readvariableop_1_resource:_
Emodel_14_separable_conv2d_42_separable_conv2d_readvariableop_resource: a
Gmodel_14_separable_conv2d_42_separable_conv2d_readvariableop_1_resource:F
8model_14_batch_normalization_104_readvariableop_resource:H
:model_14_batch_normalization_104_readvariableop_1_resource:W
Imodel_14_batch_normalization_104_fusedbatchnormv3_readvariableop_resource:Y
Kmodel_14_batch_normalization_104_fusedbatchnormv3_readvariableop_1_resource:F
8model_14_batch_normalization_102_readvariableop_resource:H
:model_14_batch_normalization_102_readvariableop_1_resource:W
Imodel_14_batch_normalization_102_fusedbatchnormv3_readvariableop_resource:Y
Kmodel_14_batch_normalization_102_fusedbatchnormv3_readvariableop_1_resource:F
8model_14_batch_normalization_100_readvariableop_resource:H
:model_14_batch_normalization_100_readvariableop_1_resource:W
Imodel_14_batch_normalization_100_fusedbatchnormv3_readvariableop_resource:Y
Kmodel_14_batch_normalization_100_fusedbatchnormv3_readvariableop_1_resource:A
.model_14_dense2_matmul_readvariableop_resource:	?=
/model_14_dense2_biasadd_readvariableop_resource:
identity??@model_14/batch_normalization_100/FusedBatchNormV3/ReadVariableOp?Bmodel_14/batch_normalization_100/FusedBatchNormV3/ReadVariableOp_1?/model_14/batch_normalization_100/ReadVariableOp?1model_14/batch_normalization_100/ReadVariableOp_1?@model_14/batch_normalization_101/FusedBatchNormV3/ReadVariableOp?Bmodel_14/batch_normalization_101/FusedBatchNormV3/ReadVariableOp_1?/model_14/batch_normalization_101/ReadVariableOp?1model_14/batch_normalization_101/ReadVariableOp_1?@model_14/batch_normalization_102/FusedBatchNormV3/ReadVariableOp?Bmodel_14/batch_normalization_102/FusedBatchNormV3/ReadVariableOp_1?/model_14/batch_normalization_102/ReadVariableOp?1model_14/batch_normalization_102/ReadVariableOp_1?@model_14/batch_normalization_103/FusedBatchNormV3/ReadVariableOp?Bmodel_14/batch_normalization_103/FusedBatchNormV3/ReadVariableOp_1?/model_14/batch_normalization_103/ReadVariableOp?1model_14/batch_normalization_103/ReadVariableOp_1?@model_14/batch_normalization_104/FusedBatchNormV3/ReadVariableOp?Bmodel_14/batch_normalization_104/FusedBatchNormV3/ReadVariableOp_1?/model_14/batch_normalization_104/ReadVariableOp?1model_14/batch_normalization_104/ReadVariableOp_1??model_14/batch_normalization_98/FusedBatchNormV3/ReadVariableOp?Amodel_14/batch_normalization_98/FusedBatchNormV3/ReadVariableOp_1?.model_14/batch_normalization_98/ReadVariableOp?0model_14/batch_normalization_98/ReadVariableOp_1??model_14/batch_normalization_99/FusedBatchNormV3/ReadVariableOp?Amodel_14/batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1?.model_14/batch_normalization_99/ReadVariableOp?0model_14/batch_normalization_99/ReadVariableOp_1?(model_14/conv2d_14/Conv2D/ReadVariableOp?&model_14/dense2/BiasAdd/ReadVariableOp?%model_14/dense2/MatMul/ReadVariableOp?5model_14/depthwise_conv2d_42/depthwise/ReadVariableOp?5model_14/depthwise_conv2d_43/depthwise/ReadVariableOp?5model_14/depthwise_conv2d_44/depthwise/ReadVariableOp?<model_14/separable_conv2d_42/separable_conv2d/ReadVariableOp?>model_14/separable_conv2d_42/separable_conv2d/ReadVariableOp_1?<model_14/separable_conv2d_43/separable_conv2d/ReadVariableOp?>model_14/separable_conv2d_43/separable_conv2d/ReadVariableOp_1?<model_14/separable_conv2d_44/separable_conv2d/ReadVariableOp?>model_14/separable_conv2d_44/separable_conv2d/ReadVariableOp_1?
(model_14/conv2d_14/Conv2D/ReadVariableOpReadVariableOp1model_14_conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:d*
dtype02*
(model_14/conv2d_14/Conv2D/ReadVariableOp?
model_14/conv2d_14/Conv2DConv2Dinput_150model_14/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_14/conv2d_14/Conv2D?
.model_14/batch_normalization_98/ReadVariableOpReadVariableOp7model_14_batch_normalization_98_readvariableop_resource*
_output_shapes
:*
dtype020
.model_14/batch_normalization_98/ReadVariableOp?
0model_14/batch_normalization_98/ReadVariableOp_1ReadVariableOp9model_14_batch_normalization_98_readvariableop_1_resource*
_output_shapes
:*
dtype022
0model_14/batch_normalization_98/ReadVariableOp_1?
?model_14/batch_normalization_98/FusedBatchNormV3/ReadVariableOpReadVariableOpHmodel_14_batch_normalization_98_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02A
?model_14/batch_normalization_98/FusedBatchNormV3/ReadVariableOp?
Amodel_14/batch_normalization_98/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJmodel_14_batch_normalization_98_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02C
Amodel_14/batch_normalization_98/FusedBatchNormV3/ReadVariableOp_1?
0model_14/batch_normalization_98/FusedBatchNormV3FusedBatchNormV3"model_14/conv2d_14/Conv2D:output:06model_14/batch_normalization_98/ReadVariableOp:value:08model_14/batch_normalization_98/ReadVariableOp_1:value:0Gmodel_14/batch_normalization_98/FusedBatchNormV3/ReadVariableOp:value:0Imodel_14/batch_normalization_98/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
is_training( 22
0model_14/batch_normalization_98/FusedBatchNormV3?
5model_14/depthwise_conv2d_44/depthwise/ReadVariableOpReadVariableOp>model_14_depthwise_conv2d_44_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype027
5model_14/depthwise_conv2d_44/depthwise/ReadVariableOp?
,model_14/depthwise_conv2d_44/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2.
,model_14/depthwise_conv2d_44/depthwise/Shape?
4model_14/depthwise_conv2d_44/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      26
4model_14/depthwise_conv2d_44/depthwise/dilation_rate?
&model_14/depthwise_conv2d_44/depthwiseDepthwiseConv2dNativeinput_15=model_14/depthwise_conv2d_44/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2(
&model_14/depthwise_conv2d_44/depthwise?
5model_14/depthwise_conv2d_43/depthwise/ReadVariableOpReadVariableOp>model_14_depthwise_conv2d_43_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype027
5model_14/depthwise_conv2d_43/depthwise/ReadVariableOp?
,model_14/depthwise_conv2d_43/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2.
,model_14/depthwise_conv2d_43/depthwise/Shape?
4model_14/depthwise_conv2d_43/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      26
4model_14/depthwise_conv2d_43/depthwise/dilation_rate?
&model_14/depthwise_conv2d_43/depthwiseDepthwiseConv2dNativeinput_15=model_14/depthwise_conv2d_43/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2(
&model_14/depthwise_conv2d_43/depthwise?
5model_14/depthwise_conv2d_42/depthwise/ReadVariableOpReadVariableOp>model_14_depthwise_conv2d_42_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype027
5model_14/depthwise_conv2d_42/depthwise/ReadVariableOp?
,model_14/depthwise_conv2d_42/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2.
,model_14/depthwise_conv2d_42/depthwise/Shape?
4model_14/depthwise_conv2d_42/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      26
4model_14/depthwise_conv2d_42/depthwise/dilation_rate?
&model_14/depthwise_conv2d_42/depthwiseDepthwiseConv2dNative4model_14/batch_normalization_98/FusedBatchNormV3:y:0=model_14/depthwise_conv2d_42/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2(
&model_14/depthwise_conv2d_42/depthwise?
/model_14/batch_normalization_103/ReadVariableOpReadVariableOp8model_14_batch_normalization_103_readvariableop_resource*
_output_shapes
:*
dtype021
/model_14/batch_normalization_103/ReadVariableOp?
1model_14/batch_normalization_103/ReadVariableOp_1ReadVariableOp:model_14_batch_normalization_103_readvariableop_1_resource*
_output_shapes
:*
dtype023
1model_14/batch_normalization_103/ReadVariableOp_1?
@model_14/batch_normalization_103/FusedBatchNormV3/ReadVariableOpReadVariableOpImodel_14_batch_normalization_103_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02B
@model_14/batch_normalization_103/FusedBatchNormV3/ReadVariableOp?
Bmodel_14/batch_normalization_103/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKmodel_14_batch_normalization_103_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02D
Bmodel_14/batch_normalization_103/FusedBatchNormV3/ReadVariableOp_1?
1model_14/batch_normalization_103/FusedBatchNormV3FusedBatchNormV3/model_14/depthwise_conv2d_44/depthwise:output:07model_14/batch_normalization_103/ReadVariableOp:value:09model_14/batch_normalization_103/ReadVariableOp_1:value:0Hmodel_14/batch_normalization_103/FusedBatchNormV3/ReadVariableOp:value:0Jmodel_14/batch_normalization_103/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
is_training( 23
1model_14/batch_normalization_103/FusedBatchNormV3?
/model_14/batch_normalization_101/ReadVariableOpReadVariableOp8model_14_batch_normalization_101_readvariableop_resource*
_output_shapes
:*
dtype021
/model_14/batch_normalization_101/ReadVariableOp?
1model_14/batch_normalization_101/ReadVariableOp_1ReadVariableOp:model_14_batch_normalization_101_readvariableop_1_resource*
_output_shapes
:*
dtype023
1model_14/batch_normalization_101/ReadVariableOp_1?
@model_14/batch_normalization_101/FusedBatchNormV3/ReadVariableOpReadVariableOpImodel_14_batch_normalization_101_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02B
@model_14/batch_normalization_101/FusedBatchNormV3/ReadVariableOp?
Bmodel_14/batch_normalization_101/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKmodel_14_batch_normalization_101_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02D
Bmodel_14/batch_normalization_101/FusedBatchNormV3/ReadVariableOp_1?
1model_14/batch_normalization_101/FusedBatchNormV3FusedBatchNormV3/model_14/depthwise_conv2d_43/depthwise:output:07model_14/batch_normalization_101/ReadVariableOp:value:09model_14/batch_normalization_101/ReadVariableOp_1:value:0Hmodel_14/batch_normalization_101/FusedBatchNormV3/ReadVariableOp:value:0Jmodel_14/batch_normalization_101/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
is_training( 23
1model_14/batch_normalization_101/FusedBatchNormV3?
.model_14/batch_normalization_99/ReadVariableOpReadVariableOp7model_14_batch_normalization_99_readvariableop_resource*
_output_shapes
:*
dtype020
.model_14/batch_normalization_99/ReadVariableOp?
0model_14/batch_normalization_99/ReadVariableOp_1ReadVariableOp9model_14_batch_normalization_99_readvariableop_1_resource*
_output_shapes
:*
dtype022
0model_14/batch_normalization_99/ReadVariableOp_1?
?model_14/batch_normalization_99/FusedBatchNormV3/ReadVariableOpReadVariableOpHmodel_14_batch_normalization_99_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02A
?model_14/batch_normalization_99/FusedBatchNormV3/ReadVariableOp?
Amodel_14/batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJmodel_14_batch_normalization_99_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02C
Amodel_14/batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1?
0model_14/batch_normalization_99/FusedBatchNormV3FusedBatchNormV3/model_14/depthwise_conv2d_42/depthwise:output:06model_14/batch_normalization_99/ReadVariableOp:value:08model_14/batch_normalization_99/ReadVariableOp_1:value:0Gmodel_14/batch_normalization_99/FusedBatchNormV3/ReadVariableOp:value:0Imodel_14/batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
is_training( 22
0model_14/batch_normalization_99/FusedBatchNormV3?
$model_14/activation_88/re_lu_88/ReluRelu5model_14/batch_normalization_103/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2&
$model_14/activation_88/re_lu_88/Relu?
$model_14/activation_86/re_lu_86/ReluRelu5model_14/batch_normalization_101/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2&
$model_14/activation_86/re_lu_86/Relu?
$model_14/activation_84/re_lu_84/ReluRelu4model_14/batch_normalization_99/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2&
$model_14/activation_84/re_lu_84/Relu?
%model_14/average_pooling2d_88/AvgPoolAvgPool2model_14/activation_88/re_lu_88/Relu:activations:0*
T0*/
_output_shapes
:?????????K*
ksize
*
paddingVALID*
strides
2'
%model_14/average_pooling2d_88/AvgPool?
%model_14/average_pooling2d_86/AvgPoolAvgPool2model_14/activation_86/re_lu_86/Relu:activations:0*
T0*/
_output_shapes
:?????????K*
ksize
*
paddingVALID*
strides
2'
%model_14/average_pooling2d_86/AvgPool?
%model_14/average_pooling2d_84/AvgPoolAvgPool2model_14/activation_84/re_lu_84/Relu:activations:0*
T0*/
_output_shapes
:?????????K*
ksize
*
paddingVALID*
strides
2'
%model_14/average_pooling2d_84/AvgPool?
model_14/dropout_88/IdentityIdentity.model_14/average_pooling2d_88/AvgPool:output:0*
T0*/
_output_shapes
:?????????K2
model_14/dropout_88/Identity?
model_14/dropout_86/IdentityIdentity.model_14/average_pooling2d_86/AvgPool:output:0*
T0*/
_output_shapes
:?????????K2
model_14/dropout_86/Identity?
model_14/dropout_84/IdentityIdentity.model_14/average_pooling2d_84/AvgPool:output:0*
T0*/
_output_shapes
:?????????K2
model_14/dropout_84/Identity?
<model_14/separable_conv2d_44/separable_conv2d/ReadVariableOpReadVariableOpEmodel_14_separable_conv2d_44_separable_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02>
<model_14/separable_conv2d_44/separable_conv2d/ReadVariableOp?
>model_14/separable_conv2d_44/separable_conv2d/ReadVariableOp_1ReadVariableOpGmodel_14_separable_conv2d_44_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype02@
>model_14/separable_conv2d_44/separable_conv2d/ReadVariableOp_1?
3model_14/separable_conv2d_44/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"   ?         25
3model_14/separable_conv2d_44/separable_conv2d/Shape?
;model_14/separable_conv2d_44/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2=
;model_14/separable_conv2d_44/separable_conv2d/dilation_rate?
7model_14/separable_conv2d_44/separable_conv2d/depthwiseDepthwiseConv2dNative%model_14/dropout_88/Identity:output:0Dmodel_14/separable_conv2d_44/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????K*
paddingSAME*
strides
29
7model_14/separable_conv2d_44/separable_conv2d/depthwise?
-model_14/separable_conv2d_44/separable_conv2dConv2D@model_14/separable_conv2d_44/separable_conv2d/depthwise:output:0Fmodel_14/separable_conv2d_44/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????K*
paddingVALID*
strides
2/
-model_14/separable_conv2d_44/separable_conv2d?
<model_14/separable_conv2d_43/separable_conv2d/ReadVariableOpReadVariableOpEmodel_14_separable_conv2d_43_separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02>
<model_14/separable_conv2d_43/separable_conv2d/ReadVariableOp?
>model_14/separable_conv2d_43/separable_conv2d/ReadVariableOp_1ReadVariableOpGmodel_14_separable_conv2d_43_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype02@
>model_14/separable_conv2d_43/separable_conv2d/ReadVariableOp_1?
3model_14/separable_conv2d_43/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"   @         25
3model_14/separable_conv2d_43/separable_conv2d/Shape?
;model_14/separable_conv2d_43/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2=
;model_14/separable_conv2d_43/separable_conv2d/dilation_rate?
7model_14/separable_conv2d_43/separable_conv2d/depthwiseDepthwiseConv2dNative%model_14/dropout_86/Identity:output:0Dmodel_14/separable_conv2d_43/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????K*
paddingSAME*
strides
29
7model_14/separable_conv2d_43/separable_conv2d/depthwise?
-model_14/separable_conv2d_43/separable_conv2dConv2D@model_14/separable_conv2d_43/separable_conv2d/depthwise:output:0Fmodel_14/separable_conv2d_43/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????K*
paddingVALID*
strides
2/
-model_14/separable_conv2d_43/separable_conv2d?
<model_14/separable_conv2d_42/separable_conv2d/ReadVariableOpReadVariableOpEmodel_14_separable_conv2d_42_separable_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02>
<model_14/separable_conv2d_42/separable_conv2d/ReadVariableOp?
>model_14/separable_conv2d_42/separable_conv2d/ReadVariableOp_1ReadVariableOpGmodel_14_separable_conv2d_42_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype02@
>model_14/separable_conv2d_42/separable_conv2d/ReadVariableOp_1?
3model_14/separable_conv2d_42/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             25
3model_14/separable_conv2d_42/separable_conv2d/Shape?
;model_14/separable_conv2d_42/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2=
;model_14/separable_conv2d_42/separable_conv2d/dilation_rate?
7model_14/separable_conv2d_42/separable_conv2d/depthwiseDepthwiseConv2dNative%model_14/dropout_84/Identity:output:0Dmodel_14/separable_conv2d_42/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????K*
paddingSAME*
strides
29
7model_14/separable_conv2d_42/separable_conv2d/depthwise?
-model_14/separable_conv2d_42/separable_conv2dConv2D@model_14/separable_conv2d_42/separable_conv2d/depthwise:output:0Fmodel_14/separable_conv2d_42/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????K*
paddingVALID*
strides
2/
-model_14/separable_conv2d_42/separable_conv2d?
/model_14/batch_normalization_104/ReadVariableOpReadVariableOp8model_14_batch_normalization_104_readvariableop_resource*
_output_shapes
:*
dtype021
/model_14/batch_normalization_104/ReadVariableOp?
1model_14/batch_normalization_104/ReadVariableOp_1ReadVariableOp:model_14_batch_normalization_104_readvariableop_1_resource*
_output_shapes
:*
dtype023
1model_14/batch_normalization_104/ReadVariableOp_1?
@model_14/batch_normalization_104/FusedBatchNormV3/ReadVariableOpReadVariableOpImodel_14_batch_normalization_104_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02B
@model_14/batch_normalization_104/FusedBatchNormV3/ReadVariableOp?
Bmodel_14/batch_normalization_104/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKmodel_14_batch_normalization_104_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02D
Bmodel_14/batch_normalization_104/FusedBatchNormV3/ReadVariableOp_1?
1model_14/batch_normalization_104/FusedBatchNormV3FusedBatchNormV36model_14/separable_conv2d_44/separable_conv2d:output:07model_14/batch_normalization_104/ReadVariableOp:value:09model_14/batch_normalization_104/ReadVariableOp_1:value:0Hmodel_14/batch_normalization_104/FusedBatchNormV3/ReadVariableOp:value:0Jmodel_14/batch_normalization_104/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????K:::::*
epsilon%o?:*
is_training( 23
1model_14/batch_normalization_104/FusedBatchNormV3?
/model_14/batch_normalization_102/ReadVariableOpReadVariableOp8model_14_batch_normalization_102_readvariableop_resource*
_output_shapes
:*
dtype021
/model_14/batch_normalization_102/ReadVariableOp?
1model_14/batch_normalization_102/ReadVariableOp_1ReadVariableOp:model_14_batch_normalization_102_readvariableop_1_resource*
_output_shapes
:*
dtype023
1model_14/batch_normalization_102/ReadVariableOp_1?
@model_14/batch_normalization_102/FusedBatchNormV3/ReadVariableOpReadVariableOpImodel_14_batch_normalization_102_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02B
@model_14/batch_normalization_102/FusedBatchNormV3/ReadVariableOp?
Bmodel_14/batch_normalization_102/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKmodel_14_batch_normalization_102_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02D
Bmodel_14/batch_normalization_102/FusedBatchNormV3/ReadVariableOp_1?
1model_14/batch_normalization_102/FusedBatchNormV3FusedBatchNormV36model_14/separable_conv2d_43/separable_conv2d:output:07model_14/batch_normalization_102/ReadVariableOp:value:09model_14/batch_normalization_102/ReadVariableOp_1:value:0Hmodel_14/batch_normalization_102/FusedBatchNormV3/ReadVariableOp:value:0Jmodel_14/batch_normalization_102/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????K:::::*
epsilon%o?:*
is_training( 23
1model_14/batch_normalization_102/FusedBatchNormV3?
/model_14/batch_normalization_100/ReadVariableOpReadVariableOp8model_14_batch_normalization_100_readvariableop_resource*
_output_shapes
:*
dtype021
/model_14/batch_normalization_100/ReadVariableOp?
1model_14/batch_normalization_100/ReadVariableOp_1ReadVariableOp:model_14_batch_normalization_100_readvariableop_1_resource*
_output_shapes
:*
dtype023
1model_14/batch_normalization_100/ReadVariableOp_1?
@model_14/batch_normalization_100/FusedBatchNormV3/ReadVariableOpReadVariableOpImodel_14_batch_normalization_100_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02B
@model_14/batch_normalization_100/FusedBatchNormV3/ReadVariableOp?
Bmodel_14/batch_normalization_100/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKmodel_14_batch_normalization_100_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02D
Bmodel_14/batch_normalization_100/FusedBatchNormV3/ReadVariableOp_1?
1model_14/batch_normalization_100/FusedBatchNormV3FusedBatchNormV36model_14/separable_conv2d_42/separable_conv2d:output:07model_14/batch_normalization_100/ReadVariableOp:value:09model_14/batch_normalization_100/ReadVariableOp_1:value:0Hmodel_14/batch_normalization_100/FusedBatchNormV3/ReadVariableOp:value:0Jmodel_14/batch_normalization_100/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????K:::::*
epsilon%o?:*
is_training( 23
1model_14/batch_normalization_100/FusedBatchNormV3?
$model_14/activation_89/re_lu_89/ReluRelu5model_14/batch_normalization_104/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????K2&
$model_14/activation_89/re_lu_89/Relu?
$model_14/activation_87/re_lu_87/ReluRelu5model_14/batch_normalization_102/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????K2&
$model_14/activation_87/re_lu_87/Relu?
$model_14/activation_85/re_lu_85/ReluRelu5model_14/batch_normalization_100/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????K2&
$model_14/activation_85/re_lu_85/Relu?
%model_14/average_pooling2d_89/AvgPoolAvgPool2model_14/activation_89/re_lu_89/Relu:activations:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2'
%model_14/average_pooling2d_89/AvgPool?
%model_14/average_pooling2d_87/AvgPoolAvgPool2model_14/activation_87/re_lu_87/Relu:activations:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2'
%model_14/average_pooling2d_87/AvgPool?
%model_14/average_pooling2d_85/AvgPoolAvgPool2model_14/activation_85/re_lu_85/Relu:activations:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2'
%model_14/average_pooling2d_85/AvgPool?
model_14/dropout_89/IdentityIdentity.model_14/average_pooling2d_89/AvgPool:output:0*
T0*/
_output_shapes
:?????????2
model_14/dropout_89/Identity?
model_14/dropout_87/IdentityIdentity.model_14/average_pooling2d_87/AvgPool:output:0*
T0*/
_output_shapes
:?????????2
model_14/dropout_87/Identity?
model_14/dropout_85/IdentityIdentity.model_14/average_pooling2d_85/AvgPool:output:0*
T0*/
_output_shapes
:?????????2
model_14/dropout_85/Identity?
model_14/flatten1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
model_14/flatten1/Const?
model_14/flatten1/ReshapeReshape%model_14/dropout_85/Identity:output:0 model_14/flatten1/Const:output:0*
T0*(
_output_shapes
:??????????2
model_14/flatten1/Reshape?
model_14/flatten64/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
model_14/flatten64/Const?
model_14/flatten64/ReshapeReshape%model_14/dropout_87/Identity:output:0!model_14/flatten64/Const:output:0*
T0*(
_output_shapes
:??????????2
model_14/flatten64/Reshape?
model_14/flatten128/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
model_14/flatten128/Const?
model_14/flatten128/ReshapeReshape%model_14/dropout_89/Identity:output:0"model_14/flatten128/Const:output:0*
T0*(
_output_shapes
:??????????2
model_14/flatten128/Reshape?
#model_14/concatenate_14/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2%
#model_14/concatenate_14/concat/axis?
model_14/concatenate_14/concatConcatV2"model_14/flatten1/Reshape:output:0#model_14/flatten64/Reshape:output:0$model_14/flatten128/Reshape:output:0,model_14/concatenate_14/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2 
model_14/concatenate_14/concat?
%model_14/dense2/MatMul/ReadVariableOpReadVariableOp.model_14_dense2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02'
%model_14/dense2/MatMul/ReadVariableOp?
model_14/dense2/MatMulMatMul'model_14/concatenate_14/concat:output:0-model_14/dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_14/dense2/MatMul?
&model_14/dense2/BiasAdd/ReadVariableOpReadVariableOp/model_14_dense2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_14/dense2/BiasAdd/ReadVariableOp?
model_14/dense2/BiasAddBiasAdd model_14/dense2/MatMul:product:0.model_14/dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_14/dense2/BiasAdd?
model_14/softmax/SoftmaxSoftmax model_14/dense2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_14/softmax/Softmax}
IdentityIdentity"model_14/softmax/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOpA^model_14/batch_normalization_100/FusedBatchNormV3/ReadVariableOpC^model_14/batch_normalization_100/FusedBatchNormV3/ReadVariableOp_10^model_14/batch_normalization_100/ReadVariableOp2^model_14/batch_normalization_100/ReadVariableOp_1A^model_14/batch_normalization_101/FusedBatchNormV3/ReadVariableOpC^model_14/batch_normalization_101/FusedBatchNormV3/ReadVariableOp_10^model_14/batch_normalization_101/ReadVariableOp2^model_14/batch_normalization_101/ReadVariableOp_1A^model_14/batch_normalization_102/FusedBatchNormV3/ReadVariableOpC^model_14/batch_normalization_102/FusedBatchNormV3/ReadVariableOp_10^model_14/batch_normalization_102/ReadVariableOp2^model_14/batch_normalization_102/ReadVariableOp_1A^model_14/batch_normalization_103/FusedBatchNormV3/ReadVariableOpC^model_14/batch_normalization_103/FusedBatchNormV3/ReadVariableOp_10^model_14/batch_normalization_103/ReadVariableOp2^model_14/batch_normalization_103/ReadVariableOp_1A^model_14/batch_normalization_104/FusedBatchNormV3/ReadVariableOpC^model_14/batch_normalization_104/FusedBatchNormV3/ReadVariableOp_10^model_14/batch_normalization_104/ReadVariableOp2^model_14/batch_normalization_104/ReadVariableOp_1@^model_14/batch_normalization_98/FusedBatchNormV3/ReadVariableOpB^model_14/batch_normalization_98/FusedBatchNormV3/ReadVariableOp_1/^model_14/batch_normalization_98/ReadVariableOp1^model_14/batch_normalization_98/ReadVariableOp_1@^model_14/batch_normalization_99/FusedBatchNormV3/ReadVariableOpB^model_14/batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1/^model_14/batch_normalization_99/ReadVariableOp1^model_14/batch_normalization_99/ReadVariableOp_1)^model_14/conv2d_14/Conv2D/ReadVariableOp'^model_14/dense2/BiasAdd/ReadVariableOp&^model_14/dense2/MatMul/ReadVariableOp6^model_14/depthwise_conv2d_42/depthwise/ReadVariableOp6^model_14/depthwise_conv2d_43/depthwise/ReadVariableOp6^model_14/depthwise_conv2d_44/depthwise/ReadVariableOp=^model_14/separable_conv2d_42/separable_conv2d/ReadVariableOp?^model_14/separable_conv2d_42/separable_conv2d/ReadVariableOp_1=^model_14/separable_conv2d_43/separable_conv2d/ReadVariableOp?^model_14/separable_conv2d_43/separable_conv2d/ReadVariableOp_1=^model_14/separable_conv2d_44/separable_conv2d/ReadVariableOp?^model_14/separable_conv2d_44/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesn
l:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2?
@model_14/batch_normalization_100/FusedBatchNormV3/ReadVariableOp@model_14/batch_normalization_100/FusedBatchNormV3/ReadVariableOp2?
Bmodel_14/batch_normalization_100/FusedBatchNormV3/ReadVariableOp_1Bmodel_14/batch_normalization_100/FusedBatchNormV3/ReadVariableOp_12b
/model_14/batch_normalization_100/ReadVariableOp/model_14/batch_normalization_100/ReadVariableOp2f
1model_14/batch_normalization_100/ReadVariableOp_11model_14/batch_normalization_100/ReadVariableOp_12?
@model_14/batch_normalization_101/FusedBatchNormV3/ReadVariableOp@model_14/batch_normalization_101/FusedBatchNormV3/ReadVariableOp2?
Bmodel_14/batch_normalization_101/FusedBatchNormV3/ReadVariableOp_1Bmodel_14/batch_normalization_101/FusedBatchNormV3/ReadVariableOp_12b
/model_14/batch_normalization_101/ReadVariableOp/model_14/batch_normalization_101/ReadVariableOp2f
1model_14/batch_normalization_101/ReadVariableOp_11model_14/batch_normalization_101/ReadVariableOp_12?
@model_14/batch_normalization_102/FusedBatchNormV3/ReadVariableOp@model_14/batch_normalization_102/FusedBatchNormV3/ReadVariableOp2?
Bmodel_14/batch_normalization_102/FusedBatchNormV3/ReadVariableOp_1Bmodel_14/batch_normalization_102/FusedBatchNormV3/ReadVariableOp_12b
/model_14/batch_normalization_102/ReadVariableOp/model_14/batch_normalization_102/ReadVariableOp2f
1model_14/batch_normalization_102/ReadVariableOp_11model_14/batch_normalization_102/ReadVariableOp_12?
@model_14/batch_normalization_103/FusedBatchNormV3/ReadVariableOp@model_14/batch_normalization_103/FusedBatchNormV3/ReadVariableOp2?
Bmodel_14/batch_normalization_103/FusedBatchNormV3/ReadVariableOp_1Bmodel_14/batch_normalization_103/FusedBatchNormV3/ReadVariableOp_12b
/model_14/batch_normalization_103/ReadVariableOp/model_14/batch_normalization_103/ReadVariableOp2f
1model_14/batch_normalization_103/ReadVariableOp_11model_14/batch_normalization_103/ReadVariableOp_12?
@model_14/batch_normalization_104/FusedBatchNormV3/ReadVariableOp@model_14/batch_normalization_104/FusedBatchNormV3/ReadVariableOp2?
Bmodel_14/batch_normalization_104/FusedBatchNormV3/ReadVariableOp_1Bmodel_14/batch_normalization_104/FusedBatchNormV3/ReadVariableOp_12b
/model_14/batch_normalization_104/ReadVariableOp/model_14/batch_normalization_104/ReadVariableOp2f
1model_14/batch_normalization_104/ReadVariableOp_11model_14/batch_normalization_104/ReadVariableOp_12?
?model_14/batch_normalization_98/FusedBatchNormV3/ReadVariableOp?model_14/batch_normalization_98/FusedBatchNormV3/ReadVariableOp2?
Amodel_14/batch_normalization_98/FusedBatchNormV3/ReadVariableOp_1Amodel_14/batch_normalization_98/FusedBatchNormV3/ReadVariableOp_12`
.model_14/batch_normalization_98/ReadVariableOp.model_14/batch_normalization_98/ReadVariableOp2d
0model_14/batch_normalization_98/ReadVariableOp_10model_14/batch_normalization_98/ReadVariableOp_12?
?model_14/batch_normalization_99/FusedBatchNormV3/ReadVariableOp?model_14/batch_normalization_99/FusedBatchNormV3/ReadVariableOp2?
Amodel_14/batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1Amodel_14/batch_normalization_99/FusedBatchNormV3/ReadVariableOp_12`
.model_14/batch_normalization_99/ReadVariableOp.model_14/batch_normalization_99/ReadVariableOp2d
0model_14/batch_normalization_99/ReadVariableOp_10model_14/batch_normalization_99/ReadVariableOp_12T
(model_14/conv2d_14/Conv2D/ReadVariableOp(model_14/conv2d_14/Conv2D/ReadVariableOp2P
&model_14/dense2/BiasAdd/ReadVariableOp&model_14/dense2/BiasAdd/ReadVariableOp2N
%model_14/dense2/MatMul/ReadVariableOp%model_14/dense2/MatMul/ReadVariableOp2n
5model_14/depthwise_conv2d_42/depthwise/ReadVariableOp5model_14/depthwise_conv2d_42/depthwise/ReadVariableOp2n
5model_14/depthwise_conv2d_43/depthwise/ReadVariableOp5model_14/depthwise_conv2d_43/depthwise/ReadVariableOp2n
5model_14/depthwise_conv2d_44/depthwise/ReadVariableOp5model_14/depthwise_conv2d_44/depthwise/ReadVariableOp2|
<model_14/separable_conv2d_42/separable_conv2d/ReadVariableOp<model_14/separable_conv2d_42/separable_conv2d/ReadVariableOp2?
>model_14/separable_conv2d_42/separable_conv2d/ReadVariableOp_1>model_14/separable_conv2d_42/separable_conv2d/ReadVariableOp_12|
<model_14/separable_conv2d_43/separable_conv2d/ReadVariableOp<model_14/separable_conv2d_43/separable_conv2d/ReadVariableOp2?
>model_14/separable_conv2d_43/separable_conv2d/ReadVariableOp_1>model_14/separable_conv2d_43/separable_conv2d/ReadVariableOp_12|
<model_14/separable_conv2d_44/separable_conv2d/ReadVariableOp<model_14/separable_conv2d_44/separable_conv2d/ReadVariableOp2?
>model_14/separable_conv2d_44/separable_conv2d/ReadVariableOp_1>model_14/separable_conv2d_44/separable_conv2d/ReadVariableOp_1:Z V
0
_output_shapes
:??????????
"
_user_specified_name
input_15
?
l
P__inference_average_pooling2d_84_layer_call_and_return_conditional_losses_588062

inputs
identity?
AvgPoolAvgPoolinputs*
T0*/
_output_shapes
:?????????K*
ksize
*
paddingVALID*
strides
2	
AvgPooll
IdentityIdentityAvgPool:output:0*
T0*/
_output_shapes
:?????????K2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
F__inference_dropout_89_layer_call_and_return_conditional_losses_585634

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_103_layer_call_and_return_conditional_losses_587934

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_98_layer_call_and_return_conditional_losses_585006

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?'
D__inference_model_14_layer_call_and_return_conditional_losses_587027

inputsB
(conv2d_14_conv2d_readvariableop_resource:d<
.batch_normalization_98_readvariableop_resource:>
0batch_normalization_98_readvariableop_1_resource:M
?batch_normalization_98_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_98_fusedbatchnormv3_readvariableop_1_resource:O
5depthwise_conv2d_44_depthwise_readvariableop_resource:O
5depthwise_conv2d_43_depthwise_readvariableop_resource:O
5depthwise_conv2d_42_depthwise_readvariableop_resource:=
/batch_normalization_103_readvariableop_resource:?
1batch_normalization_103_readvariableop_1_resource:N
@batch_normalization_103_fusedbatchnormv3_readvariableop_resource:P
Bbatch_normalization_103_fusedbatchnormv3_readvariableop_1_resource:=
/batch_normalization_101_readvariableop_resource:?
1batch_normalization_101_readvariableop_1_resource:N
@batch_normalization_101_fusedbatchnormv3_readvariableop_resource:P
Bbatch_normalization_101_fusedbatchnormv3_readvariableop_1_resource:<
.batch_normalization_99_readvariableop_resource:>
0batch_normalization_99_readvariableop_1_resource:M
?batch_normalization_99_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_99_fusedbatchnormv3_readvariableop_1_resource:W
<separable_conv2d_44_separable_conv2d_readvariableop_resource:?X
>separable_conv2d_44_separable_conv2d_readvariableop_1_resource:V
<separable_conv2d_43_separable_conv2d_readvariableop_resource:@X
>separable_conv2d_43_separable_conv2d_readvariableop_1_resource:V
<separable_conv2d_42_separable_conv2d_readvariableop_resource: X
>separable_conv2d_42_separable_conv2d_readvariableop_1_resource:=
/batch_normalization_104_readvariableop_resource:?
1batch_normalization_104_readvariableop_1_resource:N
@batch_normalization_104_fusedbatchnormv3_readvariableop_resource:P
Bbatch_normalization_104_fusedbatchnormv3_readvariableop_1_resource:=
/batch_normalization_102_readvariableop_resource:?
1batch_normalization_102_readvariableop_1_resource:N
@batch_normalization_102_fusedbatchnormv3_readvariableop_resource:P
Bbatch_normalization_102_fusedbatchnormv3_readvariableop_1_resource:=
/batch_normalization_100_readvariableop_resource:?
1batch_normalization_100_readvariableop_1_resource:N
@batch_normalization_100_fusedbatchnormv3_readvariableop_resource:P
Bbatch_normalization_100_fusedbatchnormv3_readvariableop_1_resource:8
%dense2_matmul_readvariableop_resource:	?4
&dense2_biasadd_readvariableop_resource:
identity??7batch_normalization_100/FusedBatchNormV3/ReadVariableOp?9batch_normalization_100/FusedBatchNormV3/ReadVariableOp_1?&batch_normalization_100/ReadVariableOp?(batch_normalization_100/ReadVariableOp_1?7batch_normalization_101/FusedBatchNormV3/ReadVariableOp?9batch_normalization_101/FusedBatchNormV3/ReadVariableOp_1?&batch_normalization_101/ReadVariableOp?(batch_normalization_101/ReadVariableOp_1?7batch_normalization_102/FusedBatchNormV3/ReadVariableOp?9batch_normalization_102/FusedBatchNormV3/ReadVariableOp_1?&batch_normalization_102/ReadVariableOp?(batch_normalization_102/ReadVariableOp_1?7batch_normalization_103/FusedBatchNormV3/ReadVariableOp?9batch_normalization_103/FusedBatchNormV3/ReadVariableOp_1?&batch_normalization_103/ReadVariableOp?(batch_normalization_103/ReadVariableOp_1?7batch_normalization_104/FusedBatchNormV3/ReadVariableOp?9batch_normalization_104/FusedBatchNormV3/ReadVariableOp_1?&batch_normalization_104/ReadVariableOp?(batch_normalization_104/ReadVariableOp_1?6batch_normalization_98/FusedBatchNormV3/ReadVariableOp?8batch_normalization_98/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_98/ReadVariableOp?'batch_normalization_98/ReadVariableOp_1?6batch_normalization_99/FusedBatchNormV3/ReadVariableOp?8batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_99/ReadVariableOp?'batch_normalization_99/ReadVariableOp_1?conv2d_14/Conv2D/ReadVariableOp?dense2/BiasAdd/ReadVariableOp?dense2/MatMul/ReadVariableOp?,depthwise_conv2d_42/depthwise/ReadVariableOp?,depthwise_conv2d_43/depthwise/ReadVariableOp?,depthwise_conv2d_44/depthwise/ReadVariableOp?3separable_conv2d_42/separable_conv2d/ReadVariableOp?5separable_conv2d_42/separable_conv2d/ReadVariableOp_1?3separable_conv2d_43/separable_conv2d/ReadVariableOp?5separable_conv2d_43/separable_conv2d/ReadVariableOp_1?3separable_conv2d_44/separable_conv2d/ReadVariableOp?5separable_conv2d_44/separable_conv2d/ReadVariableOp_1?
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:d*
dtype02!
conv2d_14/Conv2D/ReadVariableOp?
conv2d_14/Conv2DConv2Dinputs'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_14/Conv2D?
%batch_normalization_98/ReadVariableOpReadVariableOp.batch_normalization_98_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_98/ReadVariableOp?
'batch_normalization_98/ReadVariableOp_1ReadVariableOp0batch_normalization_98_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_98/ReadVariableOp_1?
6batch_normalization_98/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_98_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_98/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_98/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_98_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_98/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_98/FusedBatchNormV3FusedBatchNormV3conv2d_14/Conv2D:output:0-batch_normalization_98/ReadVariableOp:value:0/batch_normalization_98/ReadVariableOp_1:value:0>batch_normalization_98/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_98/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_98/FusedBatchNormV3?
,depthwise_conv2d_44/depthwise/ReadVariableOpReadVariableOp5depthwise_conv2d_44_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02.
,depthwise_conv2d_44/depthwise/ReadVariableOp?
#depthwise_conv2d_44/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2%
#depthwise_conv2d_44/depthwise/Shape?
+depthwise_conv2d_44/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2-
+depthwise_conv2d_44/depthwise/dilation_rate?
depthwise_conv2d_44/depthwiseDepthwiseConv2dNativeinputs4depthwise_conv2d_44/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
depthwise_conv2d_44/depthwise?
,depthwise_conv2d_43/depthwise/ReadVariableOpReadVariableOp5depthwise_conv2d_43_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02.
,depthwise_conv2d_43/depthwise/ReadVariableOp?
#depthwise_conv2d_43/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2%
#depthwise_conv2d_43/depthwise/Shape?
+depthwise_conv2d_43/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2-
+depthwise_conv2d_43/depthwise/dilation_rate?
depthwise_conv2d_43/depthwiseDepthwiseConv2dNativeinputs4depthwise_conv2d_43/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
depthwise_conv2d_43/depthwise?
,depthwise_conv2d_42/depthwise/ReadVariableOpReadVariableOp5depthwise_conv2d_42_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02.
,depthwise_conv2d_42/depthwise/ReadVariableOp?
#depthwise_conv2d_42/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2%
#depthwise_conv2d_42/depthwise/Shape?
+depthwise_conv2d_42/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2-
+depthwise_conv2d_42/depthwise/dilation_rate?
depthwise_conv2d_42/depthwiseDepthwiseConv2dNative+batch_normalization_98/FusedBatchNormV3:y:04depthwise_conv2d_42/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
depthwise_conv2d_42/depthwise?
&batch_normalization_103/ReadVariableOpReadVariableOp/batch_normalization_103_readvariableop_resource*
_output_shapes
:*
dtype02(
&batch_normalization_103/ReadVariableOp?
(batch_normalization_103/ReadVariableOp_1ReadVariableOp1batch_normalization_103_readvariableop_1_resource*
_output_shapes
:*
dtype02*
(batch_normalization_103/ReadVariableOp_1?
7batch_normalization_103/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_103_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_103/FusedBatchNormV3/ReadVariableOp?
9batch_normalization_103/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_103_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02;
9batch_normalization_103/FusedBatchNormV3/ReadVariableOp_1?
(batch_normalization_103/FusedBatchNormV3FusedBatchNormV3&depthwise_conv2d_44/depthwise:output:0.batch_normalization_103/ReadVariableOp:value:00batch_normalization_103/ReadVariableOp_1:value:0?batch_normalization_103/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_103/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
is_training( 2*
(batch_normalization_103/FusedBatchNormV3?
&batch_normalization_101/ReadVariableOpReadVariableOp/batch_normalization_101_readvariableop_resource*
_output_shapes
:*
dtype02(
&batch_normalization_101/ReadVariableOp?
(batch_normalization_101/ReadVariableOp_1ReadVariableOp1batch_normalization_101_readvariableop_1_resource*
_output_shapes
:*
dtype02*
(batch_normalization_101/ReadVariableOp_1?
7batch_normalization_101/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_101_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_101/FusedBatchNormV3/ReadVariableOp?
9batch_normalization_101/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_101_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02;
9batch_normalization_101/FusedBatchNormV3/ReadVariableOp_1?
(batch_normalization_101/FusedBatchNormV3FusedBatchNormV3&depthwise_conv2d_43/depthwise:output:0.batch_normalization_101/ReadVariableOp:value:00batch_normalization_101/ReadVariableOp_1:value:0?batch_normalization_101/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_101/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
is_training( 2*
(batch_normalization_101/FusedBatchNormV3?
%batch_normalization_99/ReadVariableOpReadVariableOp.batch_normalization_99_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_99/ReadVariableOp?
'batch_normalization_99/ReadVariableOp_1ReadVariableOp0batch_normalization_99_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_99/ReadVariableOp_1?
6batch_normalization_99/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_99_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_99/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_99_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_99/FusedBatchNormV3FusedBatchNormV3&depthwise_conv2d_42/depthwise:output:0-batch_normalization_99/ReadVariableOp:value:0/batch_normalization_99/ReadVariableOp_1:value:0>batch_normalization_99/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_99/FusedBatchNormV3?
activation_88/re_lu_88/ReluRelu,batch_normalization_103/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
activation_88/re_lu_88/Relu?
activation_86/re_lu_86/ReluRelu,batch_normalization_101/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
activation_86/re_lu_86/Relu?
activation_84/re_lu_84/ReluRelu+batch_normalization_99/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
activation_84/re_lu_84/Relu?
average_pooling2d_88/AvgPoolAvgPool)activation_88/re_lu_88/Relu:activations:0*
T0*/
_output_shapes
:?????????K*
ksize
*
paddingVALID*
strides
2
average_pooling2d_88/AvgPool?
average_pooling2d_86/AvgPoolAvgPool)activation_86/re_lu_86/Relu:activations:0*
T0*/
_output_shapes
:?????????K*
ksize
*
paddingVALID*
strides
2
average_pooling2d_86/AvgPool?
average_pooling2d_84/AvgPoolAvgPool)activation_84/re_lu_84/Relu:activations:0*
T0*/
_output_shapes
:?????????K*
ksize
*
paddingVALID*
strides
2
average_pooling2d_84/AvgPool?
dropout_88/IdentityIdentity%average_pooling2d_88/AvgPool:output:0*
T0*/
_output_shapes
:?????????K2
dropout_88/Identity?
dropout_86/IdentityIdentity%average_pooling2d_86/AvgPool:output:0*
T0*/
_output_shapes
:?????????K2
dropout_86/Identity?
dropout_84/IdentityIdentity%average_pooling2d_84/AvgPool:output:0*
T0*/
_output_shapes
:?????????K2
dropout_84/Identity?
3separable_conv2d_44/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_44_separable_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype025
3separable_conv2d_44/separable_conv2d/ReadVariableOp?
5separable_conv2d_44/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_44_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype027
5separable_conv2d_44/separable_conv2d/ReadVariableOp_1?
*separable_conv2d_44/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"   ?         2,
*separable_conv2d_44/separable_conv2d/Shape?
2separable_conv2d_44/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_44/separable_conv2d/dilation_rate?
.separable_conv2d_44/separable_conv2d/depthwiseDepthwiseConv2dNativedropout_88/Identity:output:0;separable_conv2d_44/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????K*
paddingSAME*
strides
20
.separable_conv2d_44/separable_conv2d/depthwise?
$separable_conv2d_44/separable_conv2dConv2D7separable_conv2d_44/separable_conv2d/depthwise:output:0=separable_conv2d_44/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????K*
paddingVALID*
strides
2&
$separable_conv2d_44/separable_conv2d?
3separable_conv2d_43/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_43_separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype025
3separable_conv2d_43/separable_conv2d/ReadVariableOp?
5separable_conv2d_43/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_43_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype027
5separable_conv2d_43/separable_conv2d/ReadVariableOp_1?
*separable_conv2d_43/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"   @         2,
*separable_conv2d_43/separable_conv2d/Shape?
2separable_conv2d_43/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_43/separable_conv2d/dilation_rate?
.separable_conv2d_43/separable_conv2d/depthwiseDepthwiseConv2dNativedropout_86/Identity:output:0;separable_conv2d_43/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????K*
paddingSAME*
strides
20
.separable_conv2d_43/separable_conv2d/depthwise?
$separable_conv2d_43/separable_conv2dConv2D7separable_conv2d_43/separable_conv2d/depthwise:output:0=separable_conv2d_43/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????K*
paddingVALID*
strides
2&
$separable_conv2d_43/separable_conv2d?
3separable_conv2d_42/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_42_separable_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype025
3separable_conv2d_42/separable_conv2d/ReadVariableOp?
5separable_conv2d_42/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_42_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype027
5separable_conv2d_42/separable_conv2d/ReadVariableOp_1?
*separable_conv2d_42/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*separable_conv2d_42/separable_conv2d/Shape?
2separable_conv2d_42/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_42/separable_conv2d/dilation_rate?
.separable_conv2d_42/separable_conv2d/depthwiseDepthwiseConv2dNativedropout_84/Identity:output:0;separable_conv2d_42/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????K*
paddingSAME*
strides
20
.separable_conv2d_42/separable_conv2d/depthwise?
$separable_conv2d_42/separable_conv2dConv2D7separable_conv2d_42/separable_conv2d/depthwise:output:0=separable_conv2d_42/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????K*
paddingVALID*
strides
2&
$separable_conv2d_42/separable_conv2d?
&batch_normalization_104/ReadVariableOpReadVariableOp/batch_normalization_104_readvariableop_resource*
_output_shapes
:*
dtype02(
&batch_normalization_104/ReadVariableOp?
(batch_normalization_104/ReadVariableOp_1ReadVariableOp1batch_normalization_104_readvariableop_1_resource*
_output_shapes
:*
dtype02*
(batch_normalization_104/ReadVariableOp_1?
7batch_normalization_104/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_104_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_104/FusedBatchNormV3/ReadVariableOp?
9batch_normalization_104/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_104_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02;
9batch_normalization_104/FusedBatchNormV3/ReadVariableOp_1?
(batch_normalization_104/FusedBatchNormV3FusedBatchNormV3-separable_conv2d_44/separable_conv2d:output:0.batch_normalization_104/ReadVariableOp:value:00batch_normalization_104/ReadVariableOp_1:value:0?batch_normalization_104/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_104/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????K:::::*
epsilon%o?:*
is_training( 2*
(batch_normalization_104/FusedBatchNormV3?
&batch_normalization_102/ReadVariableOpReadVariableOp/batch_normalization_102_readvariableop_resource*
_output_shapes
:*
dtype02(
&batch_normalization_102/ReadVariableOp?
(batch_normalization_102/ReadVariableOp_1ReadVariableOp1batch_normalization_102_readvariableop_1_resource*
_output_shapes
:*
dtype02*
(batch_normalization_102/ReadVariableOp_1?
7batch_normalization_102/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_102_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_102/FusedBatchNormV3/ReadVariableOp?
9batch_normalization_102/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_102_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02;
9batch_normalization_102/FusedBatchNormV3/ReadVariableOp_1?
(batch_normalization_102/FusedBatchNormV3FusedBatchNormV3-separable_conv2d_43/separable_conv2d:output:0.batch_normalization_102/ReadVariableOp:value:00batch_normalization_102/ReadVariableOp_1:value:0?batch_normalization_102/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_102/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????K:::::*
epsilon%o?:*
is_training( 2*
(batch_normalization_102/FusedBatchNormV3?
&batch_normalization_100/ReadVariableOpReadVariableOp/batch_normalization_100_readvariableop_resource*
_output_shapes
:*
dtype02(
&batch_normalization_100/ReadVariableOp?
(batch_normalization_100/ReadVariableOp_1ReadVariableOp1batch_normalization_100_readvariableop_1_resource*
_output_shapes
:*
dtype02*
(batch_normalization_100/ReadVariableOp_1?
7batch_normalization_100/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_100_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_100/FusedBatchNormV3/ReadVariableOp?
9batch_normalization_100/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_100_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02;
9batch_normalization_100/FusedBatchNormV3/ReadVariableOp_1?
(batch_normalization_100/FusedBatchNormV3FusedBatchNormV3-separable_conv2d_42/separable_conv2d:output:0.batch_normalization_100/ReadVariableOp:value:00batch_normalization_100/ReadVariableOp_1:value:0?batch_normalization_100/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_100/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????K:::::*
epsilon%o?:*
is_training( 2*
(batch_normalization_100/FusedBatchNormV3?
activation_89/re_lu_89/ReluRelu,batch_normalization_104/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????K2
activation_89/re_lu_89/Relu?
activation_87/re_lu_87/ReluRelu,batch_normalization_102/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????K2
activation_87/re_lu_87/Relu?
activation_85/re_lu_85/ReluRelu,batch_normalization_100/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????K2
activation_85/re_lu_85/Relu?
average_pooling2d_89/AvgPoolAvgPool)activation_89/re_lu_89/Relu:activations:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
average_pooling2d_89/AvgPool?
average_pooling2d_87/AvgPoolAvgPool)activation_87/re_lu_87/Relu:activations:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
average_pooling2d_87/AvgPool?
average_pooling2d_85/AvgPoolAvgPool)activation_85/re_lu_85/Relu:activations:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
average_pooling2d_85/AvgPool?
dropout_89/IdentityIdentity%average_pooling2d_89/AvgPool:output:0*
T0*/
_output_shapes
:?????????2
dropout_89/Identity?
dropout_87/IdentityIdentity%average_pooling2d_87/AvgPool:output:0*
T0*/
_output_shapes
:?????????2
dropout_87/Identity?
dropout_85/IdentityIdentity%average_pooling2d_85/AvgPool:output:0*
T0*/
_output_shapes
:?????????2
dropout_85/Identityq
flatten1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
flatten1/Const?
flatten1/ReshapeReshapedropout_85/Identity:output:0flatten1/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten1/Reshapes
flatten64/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
flatten64/Const?
flatten64/ReshapeReshapedropout_87/Identity:output:0flatten64/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten64/Reshapeu
flatten128/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
flatten128/Const?
flatten128/ReshapeReshapedropout_89/Identity:output:0flatten128/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten128/Reshapez
concatenate_14/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_14/concat/axis?
concatenate_14/concatConcatV2flatten1/Reshape:output:0flatten64/Reshape:output:0flatten128/Reshape:output:0#concatenate_14/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatenate_14/concat?
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense2/MatMul/ReadVariableOp?
dense2/MatMulMatMulconcatenate_14/concat:output:0$dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense2/MatMul?
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense2/BiasAdd/ReadVariableOp?
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense2/BiasAddx
softmax/SoftmaxSoftmaxdense2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
softmax/Softmaxt
IdentityIdentitysoftmax/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp8^batch_normalization_100/FusedBatchNormV3/ReadVariableOp:^batch_normalization_100/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_100/ReadVariableOp)^batch_normalization_100/ReadVariableOp_18^batch_normalization_101/FusedBatchNormV3/ReadVariableOp:^batch_normalization_101/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_101/ReadVariableOp)^batch_normalization_101/ReadVariableOp_18^batch_normalization_102/FusedBatchNormV3/ReadVariableOp:^batch_normalization_102/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_102/ReadVariableOp)^batch_normalization_102/ReadVariableOp_18^batch_normalization_103/FusedBatchNormV3/ReadVariableOp:^batch_normalization_103/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_103/ReadVariableOp)^batch_normalization_103/ReadVariableOp_18^batch_normalization_104/FusedBatchNormV3/ReadVariableOp:^batch_normalization_104/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_104/ReadVariableOp)^batch_normalization_104/ReadVariableOp_17^batch_normalization_98/FusedBatchNormV3/ReadVariableOp9^batch_normalization_98/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_98/ReadVariableOp(^batch_normalization_98/ReadVariableOp_17^batch_normalization_99/FusedBatchNormV3/ReadVariableOp9^batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_99/ReadVariableOp(^batch_normalization_99/ReadVariableOp_1 ^conv2d_14/Conv2D/ReadVariableOp^dense2/BiasAdd/ReadVariableOp^dense2/MatMul/ReadVariableOp-^depthwise_conv2d_42/depthwise/ReadVariableOp-^depthwise_conv2d_43/depthwise/ReadVariableOp-^depthwise_conv2d_44/depthwise/ReadVariableOp4^separable_conv2d_42/separable_conv2d/ReadVariableOp6^separable_conv2d_42/separable_conv2d/ReadVariableOp_14^separable_conv2d_43/separable_conv2d/ReadVariableOp6^separable_conv2d_43/separable_conv2d/ReadVariableOp_14^separable_conv2d_44/separable_conv2d/ReadVariableOp6^separable_conv2d_44/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesn
l:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2r
7batch_normalization_100/FusedBatchNormV3/ReadVariableOp7batch_normalization_100/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_100/FusedBatchNormV3/ReadVariableOp_19batch_normalization_100/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_100/ReadVariableOp&batch_normalization_100/ReadVariableOp2T
(batch_normalization_100/ReadVariableOp_1(batch_normalization_100/ReadVariableOp_12r
7batch_normalization_101/FusedBatchNormV3/ReadVariableOp7batch_normalization_101/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_101/FusedBatchNormV3/ReadVariableOp_19batch_normalization_101/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_101/ReadVariableOp&batch_normalization_101/ReadVariableOp2T
(batch_normalization_101/ReadVariableOp_1(batch_normalization_101/ReadVariableOp_12r
7batch_normalization_102/FusedBatchNormV3/ReadVariableOp7batch_normalization_102/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_102/FusedBatchNormV3/ReadVariableOp_19batch_normalization_102/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_102/ReadVariableOp&batch_normalization_102/ReadVariableOp2T
(batch_normalization_102/ReadVariableOp_1(batch_normalization_102/ReadVariableOp_12r
7batch_normalization_103/FusedBatchNormV3/ReadVariableOp7batch_normalization_103/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_103/FusedBatchNormV3/ReadVariableOp_19batch_normalization_103/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_103/ReadVariableOp&batch_normalization_103/ReadVariableOp2T
(batch_normalization_103/ReadVariableOp_1(batch_normalization_103/ReadVariableOp_12r
7batch_normalization_104/FusedBatchNormV3/ReadVariableOp7batch_normalization_104/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_104/FusedBatchNormV3/ReadVariableOp_19batch_normalization_104/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_104/ReadVariableOp&batch_normalization_104/ReadVariableOp2T
(batch_normalization_104/ReadVariableOp_1(batch_normalization_104/ReadVariableOp_12p
6batch_normalization_98/FusedBatchNormV3/ReadVariableOp6batch_normalization_98/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_98/FusedBatchNormV3/ReadVariableOp_18batch_normalization_98/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_98/ReadVariableOp%batch_normalization_98/ReadVariableOp2R
'batch_normalization_98/ReadVariableOp_1'batch_normalization_98/ReadVariableOp_12p
6batch_normalization_99/FusedBatchNormV3/ReadVariableOp6batch_normalization_99/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_99/FusedBatchNormV3/ReadVariableOp_18batch_normalization_99/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_99/ReadVariableOp%batch_normalization_99/ReadVariableOp2R
'batch_normalization_99/ReadVariableOp_1'batch_normalization_99/ReadVariableOp_12B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2>
dense2/BiasAdd/ReadVariableOpdense2/BiasAdd/ReadVariableOp2<
dense2/MatMul/ReadVariableOpdense2/MatMul/ReadVariableOp2\
,depthwise_conv2d_42/depthwise/ReadVariableOp,depthwise_conv2d_42/depthwise/ReadVariableOp2\
,depthwise_conv2d_43/depthwise/ReadVariableOp,depthwise_conv2d_43/depthwise/ReadVariableOp2\
,depthwise_conv2d_44/depthwise/ReadVariableOp,depthwise_conv2d_44/depthwise/ReadVariableOp2j
3separable_conv2d_42/separable_conv2d/ReadVariableOp3separable_conv2d_42/separable_conv2d/ReadVariableOp2n
5separable_conv2d_42/separable_conv2d/ReadVariableOp_15separable_conv2d_42/separable_conv2d/ReadVariableOp_12j
3separable_conv2d_43/separable_conv2d/ReadVariableOp3separable_conv2d_43/separable_conv2d/ReadVariableOp2n
5separable_conv2d_43/separable_conv2d/ReadVariableOp_15separable_conv2d_43/separable_conv2d/ReadVariableOp_12j
3separable_conv2d_44/separable_conv2d/ReadVariableOp3separable_conv2d_44/separable_conv2d/ReadVariableOp2n
5separable_conv2d_44/separable_conv2d/ReadVariableOp_15separable_conv2d_44/separable_conv2d/ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
l
P__inference_average_pooling2d_87_layer_call_and_return_conditional_losses_584936

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
AvgPool?
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_104_layer_call_fn_588678

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????K*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_104_layer_call_and_return_conditional_losses_5852612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????K2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????K: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
e
I__inference_activation_86_layer_call_and_return_conditional_losses_585145

inputs
identityi
re_lu_86/ReluReluinputs*
T0*0
_output_shapes
:??????????2
re_lu_86/Relux
IdentityIdentityre_lu_86/Relu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_103_layer_call_and_return_conditional_losses_584269

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_separable_conv2d_44_layer_call_and_return_conditional_losses_585204

inputsC
(separable_conv2d_readvariableop_resource:?D
*separable_conv2d_readvariableop_1_resource:
identity??separable_conv2d/ReadVariableOp?!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02!
separable_conv2d/ReadVariableOp?
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype02#
!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"   ?         2
separable_conv2d/Shape?
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rate?
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????K*
paddingSAME*
strides
2
separable_conv2d/depthwise?
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????K*
paddingVALID*
strides
2
separable_conv2d|
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*/
_output_shapes
:?????????K2

Identity?
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????K: : 2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
l
P__inference_average_pooling2d_86_layer_call_and_return_conditional_losses_588082

inputs
identity?
AvgPoolAvgPoolinputs*
T0*/
_output_shapes
:?????????K*
ksize
*
paddingVALID*
strides
2	
AvgPooll
IdentityIdentityAvgPool:output:0*
T0*/
_output_shapes
:?????????K2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
Q
5__inference_average_pooling2d_85_layer_call_fn_588741

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_85_layer_call_and_return_conditional_losses_5853622
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
?	
$__inference_signature_wrapper_586850
input_15!
unknown:d
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:#
	unknown_4:#
	unknown_5:#
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:%

unknown_19:?$

unknown_20:$

unknown_21:@$

unknown_22:$

unknown_23: $

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:	?

unknown_38:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_15unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*J
_read_only_resource_inputs,
*(	
 !"#$%&'(*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_5837232
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesn
l:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:??????????
"
_user_specified_name
input_15
?
?
S__inference_batch_normalization_101_layer_call_and_return_conditional_losses_587810

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_88_layer_call_and_return_conditional_losses_588171

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????K2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????K2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
`
D__inference_flatten1_layer_call_and_return_conditional_losses_585391

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_99_layer_call_fn_587748

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_99_layer_call_and_return_conditional_losses_5840172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
Q
5__inference_average_pooling2d_87_layer_call_fn_588761

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_87_layer_call_and_return_conditional_losses_5853562
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????K:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs
?
Q
5__inference_average_pooling2d_85_layer_call_fn_588736

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_average_pooling2d_85_layer_call_and_return_conditional_losses_5849142
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_100_layer_call_and_return_conditional_losses_588391

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????K:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????K2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????K: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????K
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
F
input_15:
serving_default_input_15:0??????????;
softmax0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?	
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer_with_weights-8
layer-18
layer_with_weights-9
layer-19
layer_with_weights-10
layer-20
layer_with_weights-11
layer-21
layer_with_weights-12
layer-22
layer_with_weights-13
layer-23
layer-24
layer-25
layer-26
layer-27
layer-28
layer-29
layer-30
 layer-31
!layer-32
"layer-33
#layer-34
$layer-35
%layer-36
&layer_with_weights-14
&layer-37
'layer-38
(	optimizer
)trainable_variables
*regularization_losses
+	variables
,	keras_api
-
signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
?

.kernel
/trainable_variables
0regularization_losses
1	variables
2	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
3axis
	4gamma
5beta
6moving_mean
7moving_variance
8trainable_variables
9regularization_losses
:	variables
;	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
<depthwise_kernel
=trainable_variables
>regularization_losses
?	variables
@	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
Adepthwise_kernel
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
Fdepthwise_kernel
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
Kaxis
	Lgamma
Mbeta
Nmoving_mean
Omoving_variance
Ptrainable_variables
Qregularization_losses
R	variables
S	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
Taxis
	Ugamma
Vbeta
Wmoving_mean
Xmoving_variance
Ytrainable_variables
Zregularization_losses
[	variables
\	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
]axis
	^gamma
_beta
`moving_mean
amoving_variance
btrainable_variables
cregularization_losses
d	variables
e	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
f
activation
gtrainable_variables
hregularization_losses
i	variables
j	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
k
activation
ltrainable_variables
mregularization_losses
n	variables
o	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
p
activation
qtrainable_variables
rregularization_losses
s	variables
t	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
utrainable_variables
vregularization_losses
w	variables
x	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
ytrainable_variables
zregularization_losses
{	variables
|	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
}trainable_variables
~regularization_losses
	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?depthwise_kernel
?pointwise_kernel
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?depthwise_kernel
?pointwise_kernel
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?depthwise_kernel
?pointwise_kernel
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?
activation
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?
activation
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?
activation
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate.m?4m?5m?<m?Am?Fm?Lm?Mm?Um?Vm?^m?_m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?.v?4v?5v?<v?Av?Fv?Lv?Mv?Uv?Vv?^v?_v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
?
.0
41
52
<3
A4
F5
L6
M7
U8
V9
^10
_11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25"
trackable_list_wrapper
 "
trackable_list_wrapper
?
.0
41
52
63
74
<5
A6
F7
L8
M9
N10
O11
U12
V13
W14
X15
^16
_17
`18
a19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39"
trackable_list_wrapper
?
?non_trainable_variables
)trainable_variables
 ?layer_regularization_losses
?metrics
*regularization_losses
+	variables
?layers
?layer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:(d2conv2d_14/kernel
'
.0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
.0"
trackable_list_wrapper
?
?non_trainable_variables
/trainable_variables
 ?layer_regularization_losses
?metrics
0regularization_losses
1	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_98/gamma
):'2batch_normalization_98/beta
2:0 (2"batch_normalization_98/moving_mean
6:4 (2&batch_normalization_98/moving_variance
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
<
40
51
62
73"
trackable_list_wrapper
?
?non_trainable_variables
8trainable_variables
 ?layer_regularization_losses
?metrics
9regularization_losses
:	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
>:<2$depthwise_conv2d_42/depthwise_kernel
'
<0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
<0"
trackable_list_wrapper
?
?non_trainable_variables
=trainable_variables
 ?layer_regularization_losses
?metrics
>regularization_losses
?	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
>:<2$depthwise_conv2d_43/depthwise_kernel
'
A0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
A0"
trackable_list_wrapper
?
?non_trainable_variables
Btrainable_variables
 ?layer_regularization_losses
?metrics
Cregularization_losses
D	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
>:<2$depthwise_conv2d_44/depthwise_kernel
'
F0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
F0"
trackable_list_wrapper
?
?non_trainable_variables
Gtrainable_variables
 ?layer_regularization_losses
?metrics
Hregularization_losses
I	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_99/gamma
):'2batch_normalization_99/beta
2:0 (2"batch_normalization_99/moving_mean
6:4 (2&batch_normalization_99/moving_variance
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
L0
M1
N2
O3"
trackable_list_wrapper
?
?non_trainable_variables
Ptrainable_variables
 ?layer_regularization_losses
?metrics
Qregularization_losses
R	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)2batch_normalization_101/gamma
*:(2batch_normalization_101/beta
3:1 (2#batch_normalization_101/moving_mean
7:5 (2'batch_normalization_101/moving_variance
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
U0
V1
W2
X3"
trackable_list_wrapper
?
?non_trainable_variables
Ytrainable_variables
 ?layer_regularization_losses
?metrics
Zregularization_losses
[	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)2batch_normalization_103/gamma
*:(2batch_normalization_103/beta
3:1 (2#batch_normalization_103/moving_mean
7:5 (2'batch_normalization_103/moving_variance
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
^0
_1
`2
a3"
trackable_list_wrapper
?
?non_trainable_variables
btrainable_variables
 ?layer_regularization_losses
?metrics
cregularization_losses
d	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
gtrainable_variables
 ?layer_regularization_losses
?metrics
hregularization_losses
i	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
ltrainable_variables
 ?layer_regularization_losses
?metrics
mregularization_losses
n	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
qtrainable_variables
 ?layer_regularization_losses
?metrics
rregularization_losses
s	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
utrainable_variables
 ?layer_regularization_losses
?metrics
vregularization_losses
w	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
ytrainable_variables
 ?layer_regularization_losses
?metrics
zregularization_losses
{	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
}trainable_variables
 ?layer_regularization_losses
?metrics
~regularization_losses
	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
>:< 2$separable_conv2d_42/depthwise_kernel
>:<2$separable_conv2d_42/pointwise_kernel
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
>:<@2$separable_conv2d_43/depthwise_kernel
>:<2$separable_conv2d_43/pointwise_kernel
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?:=?2$separable_conv2d_44/depthwise_kernel
>:<2$separable_conv2d_44/pointwise_kernel
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)2batch_normalization_100/gamma
*:(2batch_normalization_100/beta
3:1 (2#batch_normalization_100/moving_mean
7:5 (2'batch_normalization_100/moving_variance
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)2batch_normalization_102/gamma
*:(2batch_normalization_102/beta
3:1 (2#batch_normalization_102/moving_mean
7:5 (2'batch_normalization_102/moving_variance
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)2batch_normalization_104/gamma
*:(2batch_normalization_104/beta
3:1 (2#batch_normalization_104/moving_mean
7:5 (2'batch_normalization_104/moving_variance
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :	?2dense2/kernel
:2dense2/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?
60
71
N2
O3
W4
X5
`6
a7
?8
?9
?10
?11
?12
?13"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
f0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
k0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
p0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?	variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
/:-d2Adam/conv2d_14/kernel/m
/:-2#Adam/batch_normalization_98/gamma/m
.:,2"Adam/batch_normalization_98/beta/m
C:A2+Adam/depthwise_conv2d_42/depthwise_kernel/m
C:A2+Adam/depthwise_conv2d_43/depthwise_kernel/m
C:A2+Adam/depthwise_conv2d_44/depthwise_kernel/m
/:-2#Adam/batch_normalization_99/gamma/m
.:,2"Adam/batch_normalization_99/beta/m
0:.2$Adam/batch_normalization_101/gamma/m
/:-2#Adam/batch_normalization_101/beta/m
0:.2$Adam/batch_normalization_103/gamma/m
/:-2#Adam/batch_normalization_103/beta/m
C:A 2+Adam/separable_conv2d_42/depthwise_kernel/m
C:A2+Adam/separable_conv2d_42/pointwise_kernel/m
C:A@2+Adam/separable_conv2d_43/depthwise_kernel/m
C:A2+Adam/separable_conv2d_43/pointwise_kernel/m
D:B?2+Adam/separable_conv2d_44/depthwise_kernel/m
C:A2+Adam/separable_conv2d_44/pointwise_kernel/m
0:.2$Adam/batch_normalization_100/gamma/m
/:-2#Adam/batch_normalization_100/beta/m
0:.2$Adam/batch_normalization_102/gamma/m
/:-2#Adam/batch_normalization_102/beta/m
0:.2$Adam/batch_normalization_104/gamma/m
/:-2#Adam/batch_normalization_104/beta/m
%:#	?2Adam/dense2/kernel/m
:2Adam/dense2/bias/m
/:-d2Adam/conv2d_14/kernel/v
/:-2#Adam/batch_normalization_98/gamma/v
.:,2"Adam/batch_normalization_98/beta/v
C:A2+Adam/depthwise_conv2d_42/depthwise_kernel/v
C:A2+Adam/depthwise_conv2d_43/depthwise_kernel/v
C:A2+Adam/depthwise_conv2d_44/depthwise_kernel/v
/:-2#Adam/batch_normalization_99/gamma/v
.:,2"Adam/batch_normalization_99/beta/v
0:.2$Adam/batch_normalization_101/gamma/v
/:-2#Adam/batch_normalization_101/beta/v
0:.2$Adam/batch_normalization_103/gamma/v
/:-2#Adam/batch_normalization_103/beta/v
C:A 2+Adam/separable_conv2d_42/depthwise_kernel/v
C:A2+Adam/separable_conv2d_42/pointwise_kernel/v
C:A@2+Adam/separable_conv2d_43/depthwise_kernel/v
C:A2+Adam/separable_conv2d_43/pointwise_kernel/v
D:B?2+Adam/separable_conv2d_44/depthwise_kernel/v
C:A2+Adam/separable_conv2d_44/pointwise_kernel/v
0:.2$Adam/batch_normalization_100/gamma/v
/:-2#Adam/batch_normalization_100/beta/v
0:.2$Adam/batch_normalization_102/gamma/v
/:-2#Adam/batch_normalization_102/beta/v
0:.2$Adam/batch_normalization_104/gamma/v
/:-2#Adam/batch_normalization_104/beta/v
%:#	?2Adam/dense2/kernel/v
:2Adam/dense2/bias/v
?2?
D__inference_model_14_layer_call_and_return_conditional_losses_587027
D__inference_model_14_layer_call_and_return_conditional_losses_587246
D__inference_model_14_layer_call_and_return_conditional_losses_586635
D__inference_model_14_layer_call_and_return_conditional_losses_586757?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_model_14_layer_call_fn_585526
)__inference_model_14_layer_call_fn_587331
)__inference_model_14_layer_call_fn_587416
)__inference_model_14_layer_call_fn_586513?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_583723input_15"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_14_layer_call_and_return_conditional_losses_587423?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_14_layer_call_fn_587430?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_batch_normalization_98_layer_call_and_return_conditional_losses_587448
R__inference_batch_normalization_98_layer_call_and_return_conditional_losses_587466
R__inference_batch_normalization_98_layer_call_and_return_conditional_losses_587484
R__inference_batch_normalization_98_layer_call_and_return_conditional_losses_587502?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
7__inference_batch_normalization_98_layer_call_fn_587515
7__inference_batch_normalization_98_layer_call_fn_587528
7__inference_batch_normalization_98_layer_call_fn_587541
7__inference_batch_normalization_98_layer_call_fn_587554?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_depthwise_conv2d_42_layer_call_and_return_conditional_losses_587563
O__inference_depthwise_conv2d_42_layer_call_and_return_conditional_losses_587572?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_depthwise_conv2d_42_layer_call_fn_587579
4__inference_depthwise_conv2d_42_layer_call_fn_587586?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_depthwise_conv2d_43_layer_call_and_return_conditional_losses_587595
O__inference_depthwise_conv2d_43_layer_call_and_return_conditional_losses_587604?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_depthwise_conv2d_43_layer_call_fn_587611
4__inference_depthwise_conv2d_43_layer_call_fn_587618?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_depthwise_conv2d_44_layer_call_and_return_conditional_losses_587627
O__inference_depthwise_conv2d_44_layer_call_and_return_conditional_losses_587636?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_depthwise_conv2d_44_layer_call_fn_587643
4__inference_depthwise_conv2d_44_layer_call_fn_587650?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_batch_normalization_99_layer_call_and_return_conditional_losses_587668
R__inference_batch_normalization_99_layer_call_and_return_conditional_losses_587686
R__inference_batch_normalization_99_layer_call_and_return_conditional_losses_587704
R__inference_batch_normalization_99_layer_call_and_return_conditional_losses_587722?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
7__inference_batch_normalization_99_layer_call_fn_587735
7__inference_batch_normalization_99_layer_call_fn_587748
7__inference_batch_normalization_99_layer_call_fn_587761
7__inference_batch_normalization_99_layer_call_fn_587774?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_101_layer_call_and_return_conditional_losses_587792
S__inference_batch_normalization_101_layer_call_and_return_conditional_losses_587810
S__inference_batch_normalization_101_layer_call_and_return_conditional_losses_587828
S__inference_batch_normalization_101_layer_call_and_return_conditional_losses_587846?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_101_layer_call_fn_587859
8__inference_batch_normalization_101_layer_call_fn_587872
8__inference_batch_normalization_101_layer_call_fn_587885
8__inference_batch_normalization_101_layer_call_fn_587898?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_103_layer_call_and_return_conditional_losses_587916
S__inference_batch_normalization_103_layer_call_and_return_conditional_losses_587934
S__inference_batch_normalization_103_layer_call_and_return_conditional_losses_587952
S__inference_batch_normalization_103_layer_call_and_return_conditional_losses_587970?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_103_layer_call_fn_587983
8__inference_batch_normalization_103_layer_call_fn_587996
8__inference_batch_normalization_103_layer_call_fn_588009
8__inference_batch_normalization_103_layer_call_fn_588022?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_activation_84_layer_call_and_return_conditional_losses_588027?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_activation_84_layer_call_fn_588032?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_activation_86_layer_call_and_return_conditional_losses_588037?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_activation_86_layer_call_fn_588042?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_activation_88_layer_call_and_return_conditional_losses_588047?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_activation_88_layer_call_fn_588052?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_average_pooling2d_84_layer_call_and_return_conditional_losses_588057
P__inference_average_pooling2d_84_layer_call_and_return_conditional_losses_588062?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_average_pooling2d_84_layer_call_fn_588067
5__inference_average_pooling2d_84_layer_call_fn_588072?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_average_pooling2d_86_layer_call_and_return_conditional_losses_588077
P__inference_average_pooling2d_86_layer_call_and_return_conditional_losses_588082?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_average_pooling2d_86_layer_call_fn_588087
5__inference_average_pooling2d_86_layer_call_fn_588092?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_average_pooling2d_88_layer_call_and_return_conditional_losses_588097
P__inference_average_pooling2d_88_layer_call_and_return_conditional_losses_588102?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_average_pooling2d_88_layer_call_fn_588107
5__inference_average_pooling2d_88_layer_call_fn_588112?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dropout_84_layer_call_and_return_conditional_losses_588117
F__inference_dropout_84_layer_call_and_return_conditional_losses_588129?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dropout_84_layer_call_fn_588134
+__inference_dropout_84_layer_call_fn_588139?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_86_layer_call_and_return_conditional_losses_588144
F__inference_dropout_86_layer_call_and_return_conditional_losses_588156?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dropout_86_layer_call_fn_588161
+__inference_dropout_86_layer_call_fn_588166?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_88_layer_call_and_return_conditional_losses_588171
F__inference_dropout_88_layer_call_and_return_conditional_losses_588183?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dropout_88_layer_call_fn_588188
+__inference_dropout_88_layer_call_fn_588193?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_separable_conv2d_42_layer_call_and_return_conditional_losses_588205
O__inference_separable_conv2d_42_layer_call_and_return_conditional_losses_588217?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_separable_conv2d_42_layer_call_fn_588226
4__inference_separable_conv2d_42_layer_call_fn_588235?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_separable_conv2d_43_layer_call_and_return_conditional_losses_588247
O__inference_separable_conv2d_43_layer_call_and_return_conditional_losses_588259?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_separable_conv2d_43_layer_call_fn_588268
4__inference_separable_conv2d_43_layer_call_fn_588277?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_separable_conv2d_44_layer_call_and_return_conditional_losses_588289
O__inference_separable_conv2d_44_layer_call_and_return_conditional_losses_588301?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_separable_conv2d_44_layer_call_fn_588310
4__inference_separable_conv2d_44_layer_call_fn_588319?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_batch_normalization_100_layer_call_and_return_conditional_losses_588337
S__inference_batch_normalization_100_layer_call_and_return_conditional_losses_588355
S__inference_batch_normalization_100_layer_call_and_return_conditional_losses_588373
S__inference_batch_normalization_100_layer_call_and_return_conditional_losses_588391?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_100_layer_call_fn_588404
8__inference_batch_normalization_100_layer_call_fn_588417
8__inference_batch_normalization_100_layer_call_fn_588430
8__inference_batch_normalization_100_layer_call_fn_588443?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_102_layer_call_and_return_conditional_losses_588461
S__inference_batch_normalization_102_layer_call_and_return_conditional_losses_588479
S__inference_batch_normalization_102_layer_call_and_return_conditional_losses_588497
S__inference_batch_normalization_102_layer_call_and_return_conditional_losses_588515?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_102_layer_call_fn_588528
8__inference_batch_normalization_102_layer_call_fn_588541
8__inference_batch_normalization_102_layer_call_fn_588554
8__inference_batch_normalization_102_layer_call_fn_588567?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_104_layer_call_and_return_conditional_losses_588585
S__inference_batch_normalization_104_layer_call_and_return_conditional_losses_588603
S__inference_batch_normalization_104_layer_call_and_return_conditional_losses_588621
S__inference_batch_normalization_104_layer_call_and_return_conditional_losses_588639?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_104_layer_call_fn_588652
8__inference_batch_normalization_104_layer_call_fn_588665
8__inference_batch_normalization_104_layer_call_fn_588678
8__inference_batch_normalization_104_layer_call_fn_588691?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_activation_85_layer_call_and_return_conditional_losses_588696?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_activation_85_layer_call_fn_588701?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_activation_87_layer_call_and_return_conditional_losses_588706?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_activation_87_layer_call_fn_588711?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_activation_89_layer_call_and_return_conditional_losses_588716?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_activation_89_layer_call_fn_588721?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_average_pooling2d_85_layer_call_and_return_conditional_losses_588726
P__inference_average_pooling2d_85_layer_call_and_return_conditional_losses_588731?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_average_pooling2d_85_layer_call_fn_588736
5__inference_average_pooling2d_85_layer_call_fn_588741?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_average_pooling2d_87_layer_call_and_return_conditional_losses_588746
P__inference_average_pooling2d_87_layer_call_and_return_conditional_losses_588751?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_average_pooling2d_87_layer_call_fn_588756
5__inference_average_pooling2d_87_layer_call_fn_588761?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_average_pooling2d_89_layer_call_and_return_conditional_losses_588766
P__inference_average_pooling2d_89_layer_call_and_return_conditional_losses_588771?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_average_pooling2d_89_layer_call_fn_588776
5__inference_average_pooling2d_89_layer_call_fn_588781?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dropout_85_layer_call_and_return_conditional_losses_588786
F__inference_dropout_85_layer_call_and_return_conditional_losses_588798?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dropout_85_layer_call_fn_588803
+__inference_dropout_85_layer_call_fn_588808?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_87_layer_call_and_return_conditional_losses_588813
F__inference_dropout_87_layer_call_and_return_conditional_losses_588825?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dropout_87_layer_call_fn_588830
+__inference_dropout_87_layer_call_fn_588835?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_89_layer_call_and_return_conditional_losses_588840
F__inference_dropout_89_layer_call_and_return_conditional_losses_588852?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dropout_89_layer_call_fn_588857
+__inference_dropout_89_layer_call_fn_588862?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_flatten1_layer_call_and_return_conditional_losses_588868?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten1_layer_call_fn_588873?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten64_layer_call_and_return_conditional_losses_588879?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten64_layer_call_fn_588884?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_flatten128_layer_call_and_return_conditional_losses_588890?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_flatten128_layer_call_fn_588895?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_concatenate_14_layer_call_and_return_conditional_losses_588903?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_concatenate_14_layer_call_fn_588910?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense2_layer_call_and_return_conditional_losses_588920?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense2_layer_call_fn_588929?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_softmax_layer_call_and_return_conditional_losses_588934?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_softmax_layer_call_fn_588939?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_586850input_15"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_583723?<.4567FA<^_`aUVWXLMNO????????????????????:?7
0?-
+?(
input_15??????????
? "1?.
,
softmax!?
softmax??????????
I__inference_activation_84_layer_call_and_return_conditional_losses_588027j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
.__inference_activation_84_layer_call_fn_588032]8?5
.?+
)?&
inputs??????????
? "!????????????
I__inference_activation_85_layer_call_and_return_conditional_losses_588696h7?4
-?*
(?%
inputs?????????K
? "-?*
#? 
0?????????K
? ?
.__inference_activation_85_layer_call_fn_588701[7?4
-?*
(?%
inputs?????????K
? " ??????????K?
I__inference_activation_86_layer_call_and_return_conditional_losses_588037j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
.__inference_activation_86_layer_call_fn_588042]8?5
.?+
)?&
inputs??????????
? "!????????????
I__inference_activation_87_layer_call_and_return_conditional_losses_588706h7?4
-?*
(?%
inputs?????????K
? "-?*
#? 
0?????????K
? ?
.__inference_activation_87_layer_call_fn_588711[7?4
-?*
(?%
inputs?????????K
? " ??????????K?
I__inference_activation_88_layer_call_and_return_conditional_losses_588047j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
.__inference_activation_88_layer_call_fn_588052]8?5
.?+
)?&
inputs??????????
? "!????????????
I__inference_activation_89_layer_call_and_return_conditional_losses_588716h7?4
-?*
(?%
inputs?????????K
? "-?*
#? 
0?????????K
? ?
.__inference_activation_89_layer_call_fn_588721[7?4
-?*
(?%
inputs?????????K
? " ??????????K?
P__inference_average_pooling2d_84_layer_call_and_return_conditional_losses_588057?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
P__inference_average_pooling2d_84_layer_call_and_return_conditional_losses_588062i8?5
.?+
)?&
inputs??????????
? "-?*
#? 
0?????????K
? ?
5__inference_average_pooling2d_84_layer_call_fn_588067?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
5__inference_average_pooling2d_84_layer_call_fn_588072\8?5
.?+
)?&
inputs??????????
? " ??????????K?
P__inference_average_pooling2d_85_layer_call_and_return_conditional_losses_588726?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
P__inference_average_pooling2d_85_layer_call_and_return_conditional_losses_588731h7?4
-?*
(?%
inputs?????????K
? "-?*
#? 
0?????????
? ?
5__inference_average_pooling2d_85_layer_call_fn_588736?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
5__inference_average_pooling2d_85_layer_call_fn_588741[7?4
-?*
(?%
inputs?????????K
? " ???????????
P__inference_average_pooling2d_86_layer_call_and_return_conditional_losses_588077?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
P__inference_average_pooling2d_86_layer_call_and_return_conditional_losses_588082i8?5
.?+
)?&
inputs??????????
? "-?*
#? 
0?????????K
? ?
5__inference_average_pooling2d_86_layer_call_fn_588087?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
5__inference_average_pooling2d_86_layer_call_fn_588092\8?5
.?+
)?&
inputs??????????
? " ??????????K?
P__inference_average_pooling2d_87_layer_call_and_return_conditional_losses_588746?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
P__inference_average_pooling2d_87_layer_call_and_return_conditional_losses_588751h7?4
-?*
(?%
inputs?????????K
? "-?*
#? 
0?????????
? ?
5__inference_average_pooling2d_87_layer_call_fn_588756?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
5__inference_average_pooling2d_87_layer_call_fn_588761[7?4
-?*
(?%
inputs?????????K
? " ???????????
P__inference_average_pooling2d_88_layer_call_and_return_conditional_losses_588097?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
P__inference_average_pooling2d_88_layer_call_and_return_conditional_losses_588102i8?5
.?+
)?&
inputs??????????
? "-?*
#? 
0?????????K
? ?
5__inference_average_pooling2d_88_layer_call_fn_588107?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
5__inference_average_pooling2d_88_layer_call_fn_588112\8?5
.?+
)?&
inputs??????????
? " ??????????K?
P__inference_average_pooling2d_89_layer_call_and_return_conditional_losses_588766?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
P__inference_average_pooling2d_89_layer_call_and_return_conditional_losses_588771h7?4
-?*
(?%
inputs?????????K
? "-?*
#? 
0?????????
? ?
5__inference_average_pooling2d_89_layer_call_fn_588776?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
5__inference_average_pooling2d_89_layer_call_fn_588781[7?4
-?*
(?%
inputs?????????K
? " ???????????
S__inference_batch_normalization_100_layer_call_and_return_conditional_losses_588337?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_100_layer_call_and_return_conditional_losses_588355?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_100_layer_call_and_return_conditional_losses_588373v????;?8
1?.
(?%
inputs?????????K
p 
? "-?*
#? 
0?????????K
? ?
S__inference_batch_normalization_100_layer_call_and_return_conditional_losses_588391v????;?8
1?.
(?%
inputs?????????K
p
? "-?*
#? 
0?????????K
? ?
8__inference_batch_normalization_100_layer_call_fn_588404?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_100_layer_call_fn_588417?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_100_layer_call_fn_588430i????;?8
1?.
(?%
inputs?????????K
p 
? " ??????????K?
8__inference_batch_normalization_100_layer_call_fn_588443i????;?8
1?.
(?%
inputs?????????K
p
? " ??????????K?
S__inference_batch_normalization_101_layer_call_and_return_conditional_losses_587792?UVWXM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_101_layer_call_and_return_conditional_losses_587810?UVWXM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_101_layer_call_and_return_conditional_losses_587828tUVWX<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
S__inference_batch_normalization_101_layer_call_and_return_conditional_losses_587846tUVWX<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
8__inference_batch_normalization_101_layer_call_fn_587859?UVWXM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_101_layer_call_fn_587872?UVWXM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_101_layer_call_fn_587885gUVWX<?9
2?/
)?&
inputs??????????
p 
? "!????????????
8__inference_batch_normalization_101_layer_call_fn_587898gUVWX<?9
2?/
)?&
inputs??????????
p
? "!????????????
S__inference_batch_normalization_102_layer_call_and_return_conditional_losses_588461?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_102_layer_call_and_return_conditional_losses_588479?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_102_layer_call_and_return_conditional_losses_588497v????;?8
1?.
(?%
inputs?????????K
p 
? "-?*
#? 
0?????????K
? ?
S__inference_batch_normalization_102_layer_call_and_return_conditional_losses_588515v????;?8
1?.
(?%
inputs?????????K
p
? "-?*
#? 
0?????????K
? ?
8__inference_batch_normalization_102_layer_call_fn_588528?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_102_layer_call_fn_588541?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_102_layer_call_fn_588554i????;?8
1?.
(?%
inputs?????????K
p 
? " ??????????K?
8__inference_batch_normalization_102_layer_call_fn_588567i????;?8
1?.
(?%
inputs?????????K
p
? " ??????????K?
S__inference_batch_normalization_103_layer_call_and_return_conditional_losses_587916?^_`aM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_103_layer_call_and_return_conditional_losses_587934?^_`aM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_103_layer_call_and_return_conditional_losses_587952t^_`a<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
S__inference_batch_normalization_103_layer_call_and_return_conditional_losses_587970t^_`a<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
8__inference_batch_normalization_103_layer_call_fn_587983?^_`aM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_103_layer_call_fn_587996?^_`aM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_103_layer_call_fn_588009g^_`a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
8__inference_batch_normalization_103_layer_call_fn_588022g^_`a<?9
2?/
)?&
inputs??????????
p
? "!????????????
S__inference_batch_normalization_104_layer_call_and_return_conditional_losses_588585?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_104_layer_call_and_return_conditional_losses_588603?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_104_layer_call_and_return_conditional_losses_588621v????;?8
1?.
(?%
inputs?????????K
p 
? "-?*
#? 
0?????????K
? ?
S__inference_batch_normalization_104_layer_call_and_return_conditional_losses_588639v????;?8
1?.
(?%
inputs?????????K
p
? "-?*
#? 
0?????????K
? ?
8__inference_batch_normalization_104_layer_call_fn_588652?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_104_layer_call_fn_588665?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_104_layer_call_fn_588678i????;?8
1?.
(?%
inputs?????????K
p 
? " ??????????K?
8__inference_batch_normalization_104_layer_call_fn_588691i????;?8
1?.
(?%
inputs?????????K
p
? " ??????????K?
R__inference_batch_normalization_98_layer_call_and_return_conditional_losses_587448?4567M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
R__inference_batch_normalization_98_layer_call_and_return_conditional_losses_587466?4567M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
R__inference_batch_normalization_98_layer_call_and_return_conditional_losses_587484t4567<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
R__inference_batch_normalization_98_layer_call_and_return_conditional_losses_587502t4567<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
7__inference_batch_normalization_98_layer_call_fn_587515?4567M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
7__inference_batch_normalization_98_layer_call_fn_587528?4567M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
7__inference_batch_normalization_98_layer_call_fn_587541g4567<?9
2?/
)?&
inputs??????????
p 
? "!????????????
7__inference_batch_normalization_98_layer_call_fn_587554g4567<?9
2?/
)?&
inputs??????????
p
? "!????????????
R__inference_batch_normalization_99_layer_call_and_return_conditional_losses_587668?LMNOM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
R__inference_batch_normalization_99_layer_call_and_return_conditional_losses_587686?LMNOM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
R__inference_batch_normalization_99_layer_call_and_return_conditional_losses_587704tLMNO<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
R__inference_batch_normalization_99_layer_call_and_return_conditional_losses_587722tLMNO<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
7__inference_batch_normalization_99_layer_call_fn_587735?LMNOM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
7__inference_batch_normalization_99_layer_call_fn_587748?LMNOM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
7__inference_batch_normalization_99_layer_call_fn_587761gLMNO<?9
2?/
)?&
inputs??????????
p 
? "!????????????
7__inference_batch_normalization_99_layer_call_fn_587774gLMNO<?9
2?/
)?&
inputs??????????
p
? "!????????????
J__inference_concatenate_14_layer_call_and_return_conditional_losses_588903???~
w?t
r?o
#? 
inputs/0??????????
#? 
inputs/1??????????
#? 
inputs/2??????????
? "&?#
?
0??????????
? ?
/__inference_concatenate_14_layer_call_fn_588910???~
w?t
r?o
#? 
inputs/0??????????
#? 
inputs/1??????????
#? 
inputs/2??????????
? "????????????
E__inference_conv2d_14_layer_call_and_return_conditional_losses_587423m.8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
*__inference_conv2d_14_layer_call_fn_587430`.8?5
.?+
)?&
inputs??????????
? "!????????????
B__inference_dense2_layer_call_and_return_conditional_losses_588920_??0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? }
'__inference_dense2_layer_call_fn_588929R??0?-
&?#
!?
inputs??????????
? "???????????
O__inference_depthwise_conv2d_42_layer_call_and_return_conditional_losses_587563?<I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
O__inference_depthwise_conv2d_42_layer_call_and_return_conditional_losses_587572m<8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
4__inference_depthwise_conv2d_42_layer_call_fn_587579?<I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
4__inference_depthwise_conv2d_42_layer_call_fn_587586`<8?5
.?+
)?&
inputs??????????
? "!????????????
O__inference_depthwise_conv2d_43_layer_call_and_return_conditional_losses_587595?AI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
O__inference_depthwise_conv2d_43_layer_call_and_return_conditional_losses_587604mA8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
4__inference_depthwise_conv2d_43_layer_call_fn_587611?AI?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
4__inference_depthwise_conv2d_43_layer_call_fn_587618`A8?5
.?+
)?&
inputs??????????
? "!????????????
O__inference_depthwise_conv2d_44_layer_call_and_return_conditional_losses_587627?FI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
O__inference_depthwise_conv2d_44_layer_call_and_return_conditional_losses_587636mF8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
4__inference_depthwise_conv2d_44_layer_call_fn_587643?FI?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
4__inference_depthwise_conv2d_44_layer_call_fn_587650`F8?5
.?+
)?&
inputs??????????
? "!????????????
F__inference_dropout_84_layer_call_and_return_conditional_losses_588117l;?8
1?.
(?%
inputs?????????K
p 
? "-?*
#? 
0?????????K
? ?
F__inference_dropout_84_layer_call_and_return_conditional_losses_588129l;?8
1?.
(?%
inputs?????????K
p
? "-?*
#? 
0?????????K
? ?
+__inference_dropout_84_layer_call_fn_588134_;?8
1?.
(?%
inputs?????????K
p 
? " ??????????K?
+__inference_dropout_84_layer_call_fn_588139_;?8
1?.
(?%
inputs?????????K
p
? " ??????????K?
F__inference_dropout_85_layer_call_and_return_conditional_losses_588786l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
F__inference_dropout_85_layer_call_and_return_conditional_losses_588798l;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
+__inference_dropout_85_layer_call_fn_588803_;?8
1?.
(?%
inputs?????????
p 
? " ???????????
+__inference_dropout_85_layer_call_fn_588808_;?8
1?.
(?%
inputs?????????
p
? " ???????????
F__inference_dropout_86_layer_call_and_return_conditional_losses_588144l;?8
1?.
(?%
inputs?????????K
p 
? "-?*
#? 
0?????????K
? ?
F__inference_dropout_86_layer_call_and_return_conditional_losses_588156l;?8
1?.
(?%
inputs?????????K
p
? "-?*
#? 
0?????????K
? ?
+__inference_dropout_86_layer_call_fn_588161_;?8
1?.
(?%
inputs?????????K
p 
? " ??????????K?
+__inference_dropout_86_layer_call_fn_588166_;?8
1?.
(?%
inputs?????????K
p
? " ??????????K?
F__inference_dropout_87_layer_call_and_return_conditional_losses_588813l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
F__inference_dropout_87_layer_call_and_return_conditional_losses_588825l;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
+__inference_dropout_87_layer_call_fn_588830_;?8
1?.
(?%
inputs?????????
p 
? " ???????????
+__inference_dropout_87_layer_call_fn_588835_;?8
1?.
(?%
inputs?????????
p
? " ???????????
F__inference_dropout_88_layer_call_and_return_conditional_losses_588171l;?8
1?.
(?%
inputs?????????K
p 
? "-?*
#? 
0?????????K
? ?
F__inference_dropout_88_layer_call_and_return_conditional_losses_588183l;?8
1?.
(?%
inputs?????????K
p
? "-?*
#? 
0?????????K
? ?
+__inference_dropout_88_layer_call_fn_588188_;?8
1?.
(?%
inputs?????????K
p 
? " ??????????K?
+__inference_dropout_88_layer_call_fn_588193_;?8
1?.
(?%
inputs?????????K
p
? " ??????????K?
F__inference_dropout_89_layer_call_and_return_conditional_losses_588840l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
F__inference_dropout_89_layer_call_and_return_conditional_losses_588852l;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
+__inference_dropout_89_layer_call_fn_588857_;?8
1?.
(?%
inputs?????????
p 
? " ???????????
+__inference_dropout_89_layer_call_fn_588862_;?8
1?.
(?%
inputs?????????
p
? " ???????????
F__inference_flatten128_layer_call_and_return_conditional_losses_588890a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ?
+__inference_flatten128_layer_call_fn_588895T7?4
-?*
(?%
inputs?????????
? "????????????
D__inference_flatten1_layer_call_and_return_conditional_losses_588868a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ?
)__inference_flatten1_layer_call_fn_588873T7?4
-?*
(?%
inputs?????????
? "????????????
E__inference_flatten64_layer_call_and_return_conditional_losses_588879a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ?
*__inference_flatten64_layer_call_fn_588884T7?4
-?*
(?%
inputs?????????
? "????????????
D__inference_model_14_layer_call_and_return_conditional_losses_586635?<.4567FA<^_`aUVWXLMNO????????????????????B??
8?5
+?(
input_15??????????
p 

 
? "%?"
?
0?????????
? ?
D__inference_model_14_layer_call_and_return_conditional_losses_586757?<.4567FA<^_`aUVWXLMNO????????????????????B??
8?5
+?(
input_15??????????
p

 
? "%?"
?
0?????????
? ?
D__inference_model_14_layer_call_and_return_conditional_losses_587027?<.4567FA<^_`aUVWXLMNO????????????????????@?=
6?3
)?&
inputs??????????
p 

 
? "%?"
?
0?????????
? ?
D__inference_model_14_layer_call_and_return_conditional_losses_587246?<.4567FA<^_`aUVWXLMNO????????????????????@?=
6?3
)?&
inputs??????????
p

 
? "%?"
?
0?????????
? ?
)__inference_model_14_layer_call_fn_585526?<.4567FA<^_`aUVWXLMNO????????????????????B??
8?5
+?(
input_15??????????
p 

 
? "???????????
)__inference_model_14_layer_call_fn_586513?<.4567FA<^_`aUVWXLMNO????????????????????B??
8?5
+?(
input_15??????????
p

 
? "???????????
)__inference_model_14_layer_call_fn_587331?<.4567FA<^_`aUVWXLMNO????????????????????@?=
6?3
)?&
inputs??????????
p 

 
? "???????????
)__inference_model_14_layer_call_fn_587416?<.4567FA<^_`aUVWXLMNO????????????????????@?=
6?3
)?&
inputs??????????
p

 
? "???????????
O__inference_separable_conv2d_42_layer_call_and_return_conditional_losses_588205???I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
O__inference_separable_conv2d_42_layer_call_and_return_conditional_losses_588217n??7?4
-?*
(?%
inputs?????????K
? "-?*
#? 
0?????????K
? ?
4__inference_separable_conv2d_42_layer_call_fn_588226???I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
4__inference_separable_conv2d_42_layer_call_fn_588235a??7?4
-?*
(?%
inputs?????????K
? " ??????????K?
O__inference_separable_conv2d_43_layer_call_and_return_conditional_losses_588247???I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
O__inference_separable_conv2d_43_layer_call_and_return_conditional_losses_588259n??7?4
-?*
(?%
inputs?????????K
? "-?*
#? 
0?????????K
? ?
4__inference_separable_conv2d_43_layer_call_fn_588268???I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
4__inference_separable_conv2d_43_layer_call_fn_588277a??7?4
-?*
(?%
inputs?????????K
? " ??????????K?
O__inference_separable_conv2d_44_layer_call_and_return_conditional_losses_588289???I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
O__inference_separable_conv2d_44_layer_call_and_return_conditional_losses_588301n??7?4
-?*
(?%
inputs?????????K
? "-?*
#? 
0?????????K
? ?
4__inference_separable_conv2d_44_layer_call_fn_588310???I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
4__inference_separable_conv2d_44_layer_call_fn_588319a??7?4
-?*
(?%
inputs?????????K
? " ??????????K?
$__inference_signature_wrapper_586850?<.4567FA<^_`aUVWXLMNO????????????????????F?C
? 
<?9
7
input_15+?(
input_15??????????"1?.
,
softmax!?
softmax??????????
C__inference_softmax_layer_call_and_return_conditional_losses_588934X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? w
(__inference_softmax_layer_call_fn_588939K/?,
%?"
 ?
inputs?????????
? "??????????