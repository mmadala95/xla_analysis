from itertools import chain, combinations
import os
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = iterable
    return chain.from_iterable(combinations(s, t) for t in range(len(s)+1))

combinations=list(powerset(["convolution_4d_expander","gpu-conv-padding-legalization","multi_output_fusion","reduction-degenerate-dim-remover","reduction-dimension-grouper","transpose-folding"]))
print(combinations)
passes=[]
index=[]
# for num,item in enumerate(combinations):
#     print(item,num)
#
#     os.environ["XLA_FLAGS"]="--xla_disable_hlo_passes="+str.join(",",item)
#     os.environ["XLA_Current_Disabled"]="index_"+str(num);
#     os.system('python3 xla_1.py')
#     # print(os.environ["XLA_FLAGS"],os.environ["XLA_Current_Disabled"])
for num,item in enumerate(combinations):
    print(item,num)
    passes.append(str.join(",", item))
    index.append(num)
print(index,passes)

