import numpy as np
import timeit
from scipy import sparse
from sum_squares import sum_squares_cy

print('\n\n\n')
print('---------------ЗАДАНИЕ №1---------------\n')

# 1.1
float32_array = np.array([0.1, 0.2, 0.3], dtype=np.float32)
float32_exact_sum = 0.6
float32_sum = float(np.sum(float32_array))

print(
    f'Массив: {float32_array}\n'
    f'Сумма: {float32_sum}\n'
    f'Точная сумма: {float32_exact_sum}\n'
    f'Разница: {float32_sum - float32_exact_sum}\n'
)

# 1.2
eps32 = np.finfo(np.float32).eps
eps64 = np.finfo(np.float64).eps

print(
    f'Машинный эпсилон float32: {eps32}\n'
    f'Машинный эпсилон float64: {eps64}\n'
)

# 1.3
x = 1e-8

for dtype in (np.float32, np.float64):
    one = dtype(1)
    small = dtype(x)
    result = (one + small) - one
    rel_err = abs(result - x) / x

    print(
        f'{dtype.__name__}:\n'
        f'\t(1 + {x}) - 1 = {result}\n'
        f'\tТочное значение: {x}\n'
        f'\tОтносительная ошибка: {rel_err:.3e}'
    )
print('\n\n\n')
print('---------------ЗАДАНИЕ №2---------------\n')

# 2.1
matrix_a = np.arange(20).reshape(4, 5)
print(f'Все элементы второй строки: {matrix_a[1]}\n'
      f'Столбец с индексом 3: {matrix_a[:, 3]}\n'
      f'Подмассив из строк 1-3 и столбцов 2-4: \n{matrix_a[1:4, 2:5]}\n'
)

# 2.2
rows = [0, 2, 3]
columns = [1, 4]
fancy = matrix_a[np.ix_(rows, columns)]
print(f'Fancy: \n{fancy}\n')

# 2.3
non_negative = np.array([0, -1,  2, -3,  4, -5, 6, -7,  8])
non_negative[non_negative < 0] = 0
print(f'После замены отрицательных: {non_negative}\n')

# 2.4
matrix_b = np.arange(4).reshape(4, 1)
matrix_c = np.arange(5).reshape(1, 5)
print('B + C:\n', matrix_b + matrix_c)

print('\n\n\n')
print('---------------ЗАДАНИЕ №3---------------\n')

# 3.1
filename = 'big_array.dat'
shape = (1000, 1000)
mm = np.memmap(filename, dtype=np.float64, mode='w+', shape=shape)
mm[:] = np.random.rand(*shape)
print(f'Первые 5 строк и 5 столбцов сгенерированного memory-map: \n{mm[:5, :5]}\n')
mm.flush()
del mm

# 3.2
mm = np.memmap(filename, dtype=np.float64, mode='r', shape=shape)
row = mm[0, :]
col = mm[:, 0]

print(f'Первые 5 элементов 1-й строки и 1-го столбца: \n{row[:5]}\n{col[:5]}\n')

# 3.3
def copy_rows(mm):
    for i in range(mm.shape[0]):
        mm[i, :].copy()

def copy_columns(mm):
    for j in range(mm.shape[1]):
        mm[:, j].copy()


print(f'Время обхода по строкам: {timeit.timeit(lambda: copy_rows(mm), number=10)}')
print(f'Время обхода по столбцам: {timeit.timeit(lambda: copy_columns(mm), number=10)}')
print('''\nРезультат построчного обхода обычно быстрее постолбцового, 
потому что строка - это непрерывная кэш-линия, которую процессор считывает полностью. 
Во время обхода по строкам, в моменте мы "подгружаем" только одну из кэш-линий.
Во время обхода по столбцам, в моменте мы "подгружаем" всю страницу, т.е. все кэш-линии одновременно.'''
)

print('\n\n\n')
print('---------------ЗАДАНИЕ №4---------------\n')

# 4.1
S = sparse.random(5, 5, density=0.2, format='csr', random_state=42)
print(
    f'CSR:\n{S}\n'
    f'Плотный вид:\n{S.toarray()}\n'
    f'Ненулевых элементов: {S.nnz} из {5 * 5}\n'
)

# 4.2
S_coo = S.tocoo()
S_csc = S.tocsc()
print(
    f'Формат COO: {S_coo.format}\n'
    f'\tстроки: {S_coo.row}\n'
    f'\tстолбцы: {S_coo.col}\n'
    f'\tзначения: {S_coo.data}\n'
    f'Формат CSC: {S_csc.format}\n'
)

# 4.3
v = np.arange(1, 6, dtype=np.float64)
res = S @ v
print(
    f'Вектор: {v}\n'
    f'S @ v = {res}\n'
    f'Совпадает с плотным умножением: {np.allclose(res, S.toarray() @ v)}\n'
)

# 4.4
n = 1000
S_big = sparse.random(n, n, density=0.01, format='csr', random_state=42)
D_big = S_big.toarray()
v_big = np.random.rand(n)

t_sparse = min(timeit.repeat(lambda: S_big @ v_big, number=100, repeat=5)) / 100
t_dense = min(timeit.repeat(lambda: D_big @ v_big, number=100, repeat=5)) / 100
mem_sparse = S_big.data.nbytes + S_big.indices.nbytes + S_big.indptr.nbytes

print(
    f'Разреженная (CSR) @ вектор: {t_sparse:.3e} с\n'
    f'Плотная @ вектор: {t_dense:.3e} с\n'
    f'Ускорение: {t_dense / t_sparse:.1f}x\n'
    f'Память: разреженная {mem_sparse / 1e3:.0f} КБ, плотная {D_big.nbytes / 1e6:.0f} МБ\n'
)

print('\n\n\n')
print('---------------ЗАДАНИЕ №5---------------\n')

# 5.1
def sum_squares_py(lst):
    total = 0.0
    for x in lst:
        total += x * x
    return total

# 5.2 (функция sum_squares_cy импортирована из скомпилированного модуля)


# 5.3
n = 10 ** 7
arr = np.random.rand(n)
lst = arr.tolist()

res_py = sum_squares_py(lst)
res_cy = sum_squares_cy(arr)

t_py = timeit.timeit(lambda: sum_squares_py(lst), number=3) / 3
t_cy = timeit.timeit(lambda: sum_squares_cy(arr), number=3) / 3
t_np = timeit.timeit(lambda: np.dot(arr, arr), number=3) / 3

print(
    f'Python: {res_py}\n'
    f'Cython: {res_cy}\n'
    f'Результаты совпадают: {np.isclose(res_py, res_cy)}\n\n'
    f'Время Python: {t_py:.4f} с\n'
    f'Время Cython: {t_cy:.4f} с\n'
    f'Время NumPy (np.dot): {t_np:.4f} с\n'
    f'Ускорение Cython относительно Python: {t_py / t_cy:.1f}x\n'
)

