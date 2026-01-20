# dataset
用nC表示通道数

添加get_shifts函数，phase_cross_correlation函数计算平移大小

_getitem_ 按数组平移，相乘，叠加

返回值第三个值改为label_shift，平移后的gt数据

# MST
只改了class MST:

用nC表示通道数

initial_x修改为平移的逆操作

# train_v1
# test_check
