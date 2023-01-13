#B問題#
N,M,T = map(int,input().split())
A_l = list(map(int,input().split()))
xy_l = []
# s = input()
for j in range(M):
    xy_l.append(input().split())#こうしないとYiにアクセスできなかった
num = 0#部屋初期値
a = []#移動に成功したら#を入れる
l = len(A_l)#部屋数
l2 = len(xy_l)#ボーナス部屋の数=M
while num < l:
    num += 1
    if num < M+1:
        # print(888888888888)
        if num == int(xy_l[num-1][0])-int(1):
            # print(T)
            # break
            T = T + int(xy_l[num-1][1])
            #print(T)
    #print('#####',num)
    if  T - A_l[num-1] > 0:
        T = T - A_l[num-1]
        #print(T)
        # print(int(xy_l[num-1][1])-int(1))
        # print(type(int(xy_l[num-1][1])))
        # break
        #注:ボーナス部屋以下なら(これないと35行目でlist out of rangeのエラー)
        # if num < M+1:
        #     if num == int(xy_l[num-1][0])-int(1):
        #         # print(T)
        #         # break
        #         T = T + int(xy_l[num-1][1])
        #         #print(T)
        a.append('#')
    else:
        break
#print(a)
if len(a) == l:
    print('Yes')
else:
    print('No')