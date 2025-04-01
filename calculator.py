# 导入所有必要的库（整合并排序导入）
import math
import re
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np
import scipy
import scipy.integrate
import scipy.interpolate
import scipy.optimize
import scipy.special
import scipy.stats
from mcp.server.fastmcp import FastMCP

# 设置默认编码为UTF-8
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# 初始化 MCP 服务器
app = FastMCP("calculator")


# ===== 安全计算环境 =====
class SafeEvaluator:
    """安全表达式求值器"""

    def __init__(self) -> None:
        from math import acos, asin, atan, cos, cosh, e, exp, fabs, factorial
        from math import log, log10, pi, sin, sinh, sqrt, tan, tanh

        self.allowed_names: Dict[str, Any] = {
            # 三角函数
            'sin': sin, 'cos': cos, 'tan': tan,
            'asin': asin, 'acos': acos, 'atan': atan,
            # 双曲函数
            'sinh': sinh, 'cosh': cosh, 'tanh': tanh,
            # 指数对数
            'exp': exp, 'log': log, 'log10': log10, 'ln': log,
            # 其它数学函数
            'sqrt': sqrt, 'abs': fabs, 'factorial': factorial,
            # 常量
            'pi': pi, 'e': e
        }

    def eval(self, expr: str, variables: Optional[Dict[str, float]] = None) -> Any:
        """安全计算表达式"""
        variables = variables or {}

        # 安全性验证
        for key in variables:
            if not isinstance(key, str) or not key.isalpha():
                raise ValueError(f"非法变量名: {key}")

        for value in variables.values():
            if not isinstance(value, (int, float, complex)):
                raise ValueError(f"变量值必须是数值类型")

        # 替换常见的数学符号
        expr = expr.replace('^', '**')  # 幂运算
        expr = expr.replace('÷', '/')  # 除法
        expr = expr.replace('×', '*')  # 乘法

        # 支持阶乘符号
        factorial_pattern = r'(\d+)!'
        for match in re.finditer(factorial_pattern, expr):
            num = match.group(1)
            expr = expr.replace(f"{num}!", f"factorial({num})")

        # 构建安全命名空间
        namespace = {**self.allowed_names, **variables}

        try:
            # 使用eval安全计算表达式
            result = eval(expr, {'__builtins__': None}, namespace)

            # 处理返回结果
            if isinstance(result, (int, float, complex, list, tuple, np.ndarray)):
                if isinstance(result, np.ndarray):
                    return result.tolist()
                return result
            else:
                raise TypeError(f"表达式结果类型不支持: {type(result)}")
        except Exception as e:
            raise ValueError(f"表达式计算错误: {str(e)}")


# 全局安全评估器实例
evaluator = SafeEvaluator()


# ===== 1. 基础算术运算 =====
@app.tool(name="add", description="加法：a+b")
def add(a: float, b: float) -> float:
    """计算两数之和"""
    return float(np.add(a, b))


@app.tool(name="subtract", description="减法：a-b")
def subtract(a: float, b: float) -> float:
    """计算两数之差"""
    return float(np.subtract(a, b))


@app.tool(name="multiply", description="乘法：a×b")
def multiply(a: float, b: float) -> float:
    """计算两数之积"""
    return float(np.multiply(a, b))


@app.tool(name="divide", description="除法：a÷b (b≠0)")
def divide(a: float, b: float) -> float:
    """计算两数之商"""
    if abs(b) < 1e-15:
        raise ValueError("除数不能为零")
    return float(np.divide(a, b))


@app.tool(name="power", description="幂运算：a^b")
def power(a: float, b: float) -> float:
    """计算a的b次方"""
    return float(np.power(a, b))


@app.tool(name="sqrt", description="平方根：√a (a≥0)")
def sqrt_fn(a: float) -> float:
    """计算平方根"""
    if a < 0:
        raise ValueError("不能计算负数的平方根")
    return float(np.sqrt(a))


@app.tool(name="mod", description="取模/余数：a mod b")
def mod(a: float, b: float) -> float:
    """计算a除以b的余数"""
    if abs(b) < 1e-15:
        raise ValueError("除数不能为零")
    return float(np.mod(a, b))


# ===== 2. 高级数学函数 =====
@app.tool(name="exp", description="指数函数：e^x")
def exp_fn(x: float) -> float:
    """计算e的x次方"""
    return float(np.exp(x))


@app.tool(name="log", description="自然对数：ln(x) (x>0)")
def log_fn(x: float) -> float:
    """计算自然对数"""
    if x <= 0:
        raise ValueError("对数函数参数必须为正数")
    return float(np.log(x))


@app.tool(name="log10", description="常用对数：log10(x) (x>0)")
def log10_fn(x: float) -> float:
    """计算以10为底的对数"""
    if x <= 0:
        raise ValueError("对数函数参数必须为正数")
    return float(np.log10(x))


@app.tool(name="log_base", description="任意底对数：log_b(x) (x>0, b>0, b≠1)")
def log_base_fn(x: float, base: float) -> float:
    """计算任意底对数"""
    if x <= 0 or base <= 0 or abs(base - 1) < 1e-15:
        raise ValueError("参数无效：x和base必须为正数，且base≠1")
    return float(np.log(x) / np.log(base))


# ===== 3. A. 三角函数 =====
@app.tool(name="sin", description="正弦函数：sin(x)")
def sin_fn(x: float) -> float:
    """计算正弦值（弧度制）"""
    return float(np.sin(x))


@app.tool(name="cos", description="余弦函数：cos(x)")
def cos_fn(x: float) -> float:
    """计算余弦值（弧度制）"""
    return float(np.cos(x))


@app.tool(name="tan", description="正切函数：tan(x)")
def tan_fn(x: float) -> float:
    """计算正切值（弧度制）"""
    return float(np.tan(x))


# ===== 3. B. 反三角函数 =====
@app.tool(name="asin", description="反正弦函数：arcsin(x) (-1≤x≤1)")
def asin_fn(x: float) -> float:
    """计算反正弦值（弧度制）"""
    if x < -1 or x > 1:
        raise ValueError("反正弦函数的参数必须在[-1,1]范围内")
    return float(np.arcsin(x))


@app.tool(name="acos", description="反余弦函数：arccos(x) (-1≤x≤1)")
def acos_fn(x: float) -> float:
    """计算反余弦值（弧度制）"""
    if x < -1 or x > 1:
        raise ValueError("反余弦函数的参数必须在[-1,1]范围内")
    return float(np.arccos(x))


@app.tool(name="atan", description="反正切函数：arctan(x)")
def atan_fn(x: float) -> float:
    """计算反正切值（弧度制）"""
    return float(np.arctan(x))


@app.tool(name="atan2", description="二参数反正切函数：arctan2(y,x)")
def atan2_fn(y: float, x: float) -> float:
    """计算二参数反正切值（弧度制）"""
    return float(np.arctan2(y, x))


# ===== 4. 双曲函数 =====
@app.tool(name="sinh", description="双曲正弦：sinh(x)")
def sinh_fn(x: float) -> float:
    """计算双曲正弦值"""
    return float(np.sinh(x))


@app.tool(name="cosh", description="双曲余弦：cosh(x)")
def cosh_fn(x: float) -> float:
    """计算双曲余弦值"""
    return float(np.cosh(x))


@app.tool(name="tanh", description="双曲正切：tanh(x)")
def tanh_fn(x: float) -> float:
    """计算双曲正切值"""
    return float(np.tanh(x))


# ===== 5. 角度转换 =====
@app.tool(name="rad2deg", description="弧度转角度：rad→deg")
def rad2deg(rad: float) -> float:
    """将弧度转换为角度"""
    return float(np.rad2deg(rad))


@app.tool(name="deg2rad", description="角度转弧度：deg→rad")
def deg2rad(deg: float) -> float:
    """将角度转换为弧度"""
    return float(np.deg2rad(deg))


# ===== 6. 向量运算 =====
@app.tool(name="vector_add", description="向量加法：v1+v2")
def vector_add(v1: List[float], v2: List[float]) -> List[float]:
    """计算两个向量的和"""
    if len(v1) != len(v2):
        raise ValueError("向量维度必须相同")
    return [float(a + b) for a, b in zip(v1, v2)]


@app.tool(name="vector_subtract", description="向量减法：v1-v2")
def vector_subtract(v1: List[float], v2: List[float]) -> List[float]:
    """计算两个向量的差"""
    if len(v1) != len(v2):
        raise ValueError("向量维度必须相同")
    return [float(a - b) for a, b in zip(v1, v2)]


@app.tool(name="vector_dot", description="向量点积：v1·v2")
def vector_dot(v1: List[float], v2: List[float]) -> float:
    """计算两个向量的点积"""
    if len(v1) != len(v2):
        raise ValueError("向量维度必须相同")
    return float(np.dot(v1, v2))


@app.tool(name="vector_cross", description="向量叉积：v1×v2 (仅限3D向量)")
def vector_cross(v1: List[float], v2: List[float]) -> List[float]:
    """计算两个3D向量的叉积"""
    if len(v1) != 3 or len(v2) != 3:
        raise ValueError("叉积仅适用于3维向量")
    result = np.cross(v1, v2)
    return [float(x) for x in result]


@app.tool(name="vector_norm", description="向量范数：|v|")
def vector_norm(v: List[float], ord: Optional[int] = 2) -> float:
    """计算向量的范数，默认为欧几里得范数(L2)"""
    return float(np.linalg.norm(v, ord=ord))


@app.tool(name="vector_angle", description="向量夹角(弧度)：∠(v1,v2)")
def vector_angle(v1: List[float], v2: List[float]) -> float:
    """计算两个向量之间的夹角（弧度）"""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-15 or norm2 < 1e-15:
        raise ValueError("不能计算零向量的夹角")
    cos_angle = dot / (norm1 * norm2)
    # 处理数值误差导致的范围溢出
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    return float(np.arccos(cos_angle))


# ===== 7. 矩阵基础运算 =====
@app.tool(name="matrix_add", description="矩阵加法：A+B")
def matrix_add(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """计算两个矩阵的和"""
    try:
        A_np = np.array(A, dtype=np.float64)
        B_np = np.array(B, dtype=np.float64)
        if A_np.shape != B_np.shape:
            raise ValueError(f"矩阵维度不匹配：{A_np.shape} vs {B_np.shape}")
        result = np.add(A_np, B_np)
        return result.tolist()
    except Exception as e:
        raise ValueError(f"矩阵加法计算失败: {str(e)}")


@app.tool(name="matrix_subtract", description="矩阵减法：A-B")
def matrix_subtract(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """计算两个矩阵的差"""
    try:
        A_np = np.array(A, dtype=np.float64)
        B_np = np.array(B, dtype=np.float64)
        if A_np.shape != B_np.shape:
            raise ValueError(f"矩阵维度不匹配：{A_np.shape} vs {B_np.shape}")
        result = np.subtract(A_np, B_np)
        return result.tolist()
    except Exception as e:
        raise ValueError(f"矩阵减法计算失败: {str(e)}")


@app.tool(name="matrix_multiply", description="矩阵乘法：A×B")
def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """计算两个矩阵的乘积"""
    try:
        A_np = np.array(A, dtype=np.float64)
        B_np = np.array(B, dtype=np.float64)
        if A_np.shape[1] != B_np.shape[0]:
            raise ValueError(f"矩阵维度不兼容：{A_np.shape} × {B_np.shape}")
        result = np.matmul(A_np, B_np)
        return result.tolist()
    except Exception as e:
        raise ValueError(f"矩阵乘法计算失败: {str(e)}")


@app.tool(name="matrix_transpose", description="矩阵转置：A^T")
def matrix_transpose(A: List[List[float]]) -> List[List[float]]:
    """计算矩阵的转置"""
    try:
        A_np = np.array(A, dtype=np.float64)
        result = A_np.T
        return result.tolist()
    except Exception as e:
        raise ValueError(f"矩阵转置计算失败: {str(e)}")


# ===== 8. 矩阵高级运算 =====
@app.tool(name="matrix_inverse", description="矩阵求逆：A^(-1)")
def matrix_inverse(A: List[List[float]]) -> List[List[float]]:
    """计算矩阵的逆"""
    try:
        A_np = np.array(A, dtype=np.float64)
        if A_np.shape[0] != A_np.shape[1]:
            raise ValueError("只能求方阵的逆")
        inv = np.linalg.inv(A_np)
        return inv.tolist()
    except np.linalg.LinAlgError:
        raise ValueError("矩阵不可逆")
    except Exception as e:
        raise ValueError(f"矩阵求逆计算失败: {str(e)}")


@app.tool(name="matrix_determinant", description="矩阵行列式：det(A)")
def matrix_determinant(A: List[List[float]]) -> float:
    """计算矩阵的行列式"""
    try:
        A_np = np.array(A, dtype=np.float64)
        if A_np.shape[0] != A_np.shape[1]:
            raise ValueError("只能计算方阵的行列式")
        return float(np.linalg.det(A_np))
    except Exception as e:
        raise ValueError(f"行列式计算失败: {str(e)}")


@app.tool(name="matrix_rank", description="矩阵的秩：rank(A)")
def matrix_rank(A: List[List[float]]) -> int:
    """计算矩阵的秩"""
    try:
        A_np = np.array(A, dtype=np.float64)
        return int(np.linalg.matrix_rank(A_np))
    except Exception as e:
        raise ValueError(f"矩阵秩计算失败: {str(e)}")


@app.tool(name="matrix_trace", description="矩阵的迹：tr(A)")
def matrix_trace(A: List[List[float]]) -> float:
    """计算矩阵的迹(对角线元素之和)"""
    try:
        A_np = np.array(A, dtype=np.float64)
        if A_np.shape[0] != A_np.shape[1]:
            raise ValueError("只能计算方阵的迹")
        return float(np.trace(A_np))
    except Exception as e:
        raise ValueError(f"矩阵迹计算失败: {str(e)}")


@app.tool(name="matrix_eigenvalues", description="矩阵特征值：eig(A)")
def matrix_eigenvalues(A: List[List[float]]) -> List[complex]:
    """计算矩阵的特征值"""
    try:
        A_np = np.array(A, dtype=np.float64)
        if A_np.shape[0] != A_np.shape[1]:
            raise ValueError("只能计算方阵的特征值")
        eigenvals = np.linalg.eigvals(A_np)
        return [complex(val.real, val.imag) for val in eigenvals]
    except Exception as e:
        raise ValueError(f"特征值计算失败: {str(e)}")


@app.tool(name="matrix_eigenvectors", description="矩阵特征向量：eigvec(A)")
def matrix_eigenvectors(A: List[List[float]]) -> Dict[str, Any]:
    """计算矩阵的特征值和特征向量"""
    try:
        A_np = np.array(A, dtype=np.float64)
        if A_np.shape[0] != A_np.shape[1]:
            raise ValueError("只能计算方阵的特征向量")
        eigenvals, eigenvecs = np.linalg.eig(A_np)

        # 转换为更友好的格式
        eigenvalues = [complex(val.real, val.imag) for val in eigenvals]
        eigenvectors = []
        for i in range(eigenvecs.shape[1]):
            vector = eigenvecs[:, i]
            eigenvectors.append([complex(val.real, val.imag) for val in vector])

        return {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors
        }
    except Exception as e:
        raise ValueError(f"特征向量计算失败: {str(e)}")


# ===== 9. 线性方程组求解 =====
@app.tool(name="solve_linear_system", description="解线性方程组：Ax=b")
def solve_linear_system(A: List[List[float]], b: List[float]) -> Dict[str, Any]:
    """求解线性方程组 Ax = b"""
    try:
        A_np = np.array(A, dtype=np.float64)
        b_np = np.array(b, dtype=np.float64)

        # 确保维度匹配
        if A_np.shape[0] != len(b_np):
            raise ValueError(f"矩阵A的行数必须等于向量b的长度: {A_np.shape[0]} vs {len(b_np)}")

        # 尝试直接求解
        try:
            x = np.linalg.solve(A_np, b_np)
            return {
                "solution": x.tolist(),
                "method": "direct",
                "residual_norm": float(np.linalg.norm(A_np @ x - b_np))
            }
        except np.linalg.LinAlgError:
            # 尝试最小二乘解
            x, residuals, rank, s = np.linalg.lstsq(A_np, b_np, rcond=None)
            return {
                "solution": x.tolist(),
                "method": "least_squares",
                "rank": int(rank),
                "singular_values": s.tolist(),
                "residual_norm": float(np.linalg.norm(A_np @ x - b_np))
            }
    except Exception as e:
        raise ValueError(f"线性方程组求解失败: {str(e)}")


# ===== 10. 微分运算 =====
@app.tool(name="numerical_derivative", description="数值导数：f'(x)")
def numerical_derivative(f: str, x: float, h: float = 1e-6) -> float:
    """计算函数在指定点的导数"""
    try:
        # 创建安全的数学环境
        math_env = evaluator.allowed_names.copy()

        # 清理表达式
        f = f.replace('^', '**')

        # 创建函数
        def func(t: float) -> float:
            local_vars = {'x': t, **math_env}
            return eval(f, {'__builtins__': None}, local_vars)

        # 中心差分法
        result = (func(x + h) - func(x - h)) / (2 * h)
        return float(result)
    except Exception as e:
        raise ValueError(f"导数计算失败: {str(e)}")


@app.tool(name="higher_derivative", description="高阶导数：f^(n)(x)")
def higher_derivative(f: str, x: float, order: int = 1, h: float = 1e-5) -> float:
    """计算函数在指定点的n阶导数"""
    try:
        if order < 1:
            raise ValueError("导数阶数必须为正整数")

        # 创建安全的数学环境
        math_env = evaluator.allowed_names.copy()

        # 清理表达式
        f = f.replace('^', '**')

        # 创建函数
        def func(t: float) -> float:
            local_vars = {'x': t, **math_env}
            return eval(f, {'__builtins__': None}, local_vars)

        # 使用scipy的导数函数
        from scipy.misc import derivative
        result = derivative(func, x, dx=h, n=order, order=max(2 * order + 1, 3))
        return float(result)
    except Exception as e:
        raise ValueError(f"高阶导数计算失败: {str(e)}")


# ===== 11. 积分运算 =====
@app.tool(name="definite_integral", description="定积分：∫[a,b] f(x) dx")
def definite_integral(f: str, a: float, b: float, tol: float = 1e-8) -> Dict[str, float]:
    """计算函数在指定区间上的定积分"""
    try:
        # 创建安全的数学环境
        math_env = evaluator.allowed_names.copy()

        # 清理表达式
        f = f.replace('^', '**')

        # 创建函数
        def func(t: float) -> float:
            local_vars = {'x': t, **math_env}
            return eval(f, {'__builtins__': None}, local_vars)

        # 使用scipy的积分函数
        from scipy.integrate import quad
        result, error = quad(func, a, b, epsabs=tol, epsrel=tol)

        return {
            "integral_value": float(result),
            "estimated_error": float(error)
        }
    except Exception as e:
        raise ValueError(f"定积分计算失败: {str(e)}")


@app.tool(name="improper_integral", description="无穷积分：∫[a,∞) f(x) dx 或 ∫(-∞,b] f(x) dx")
def improper_integral(
        f: str,
        a: Optional[float] = None,
        b: Optional[float] = None,
        tol: float = 1e-6
) -> Dict[str, float]:
    """计算函数在无穷区间上的积分"""
    try:
        if a is None and b is None:
            raise ValueError("至少需要提供一个积分上下限")

        # 创建安全的数学环境
        math_env = evaluator.allowed_names.copy()

        # 清理表达式
        f = f.replace('^', '**')

        # 创建函数
        def func(t: float) -> float:
            local_vars = {'x': t, **math_env}
            return eval(f, {'__builtins__': None}, local_vars)

        # 使用scipy的积分函数
        from scipy.integrate import quad
        result, error = quad(
            func,
            a if a is not None else -np.inf,
            b if b is not None else np.inf,
            epsabs=tol,
            epsrel=tol
        )

        return {
            "integral_value": float(result),
            "estimated_error": float(error)
        }
    except Exception as e:
        raise ValueError(f"无穷积分计算失败: {str(e)}")


# ===== 12. 统计运算 =====
@app.tool(name="descriptive_stats", description="描述性统计：计算均值、方差等")
def descriptive_stats(data: List[float]) -> Dict[str, float]:
    """计算数据的描述性统计量"""
    try:
        if not data:
            raise ValueError("数据不能为空")

        data_np = np.array(data, dtype=np.float64)

        return {
            "count": len(data),
            "mean": float(np.mean(data_np)),
            "median": float(np.median(data_np)),
            "std": float(np.std(data_np, ddof=1)),
            "var": float(np.var(data_np, ddof=1)),
            "min": float(np.min(data_np)),
            "max": float(np.max(data_np)),
            "range": float(np.max(data_np) - np.min(data_np)),
            "q1": float(np.percentile(data_np, 25)),
            "q3": float(np.percentile(data_np, 75))
        }
    except Exception as e:
        raise ValueError(f"统计计算失败: {str(e)}")


@app.tool(name="correlation", description="相关性分析：计算相关系数")
def correlation(x: List[float], y: List[float]) -> Dict[str, float]:
    """计算两组数据间的相关性"""
    try:
        if len(x) != len(y):
            raise ValueError("两组数据长度必须相同")

        if len(x) < 2:
            raise ValueError("数据点数量必须大于1")

        x_np = np.array(x, dtype=np.float64)
        y_np = np.array(y, dtype=np.float64)

        pearson_r, p_value = scipy.stats.pearsonr(x_np, y_np)

        return {
            "pearson_r": float(pearson_r),
            "p_value": float(p_value),
            "covariance": float(np.cov(x_np, y_np)[0, 1])
        }
    except Exception as e:
        raise ValueError(f"相关性计算失败: {str(e)}")


# ===== 13. 数据拟合与插值 =====
@app.tool(name="linear_fit", description="线性拟合：y = ax + b")
def linear_fit(x: List[float], y: List[float]) -> Dict[str, Union[float, str]]:
    """计算最佳线性拟合参数"""
    try:
        if len(x) != len(y):
            raise ValueError("x和y数据点数量必须相同")

        if len(x) < 2:
            raise ValueError("至少需要2个数据点进行拟合")

        x_np = np.array(x, dtype=np.float64)
        y_np = np.array(y, dtype=np.float64)

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x_np, y_np)

        # 计算预测值和残差
        y_pred = slope * x_np + intercept
        residuals = y_np - y_pred

        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_value ** 2),
            "p_value": float(p_value),
            "std_err": float(std_err),
            "equation": f"y = {slope:.6g}x + {intercept:.6g}"
        }
    except Exception as e:
        raise ValueError(f"线性拟合失败: {str(e)}")


@app.tool(name="polynomial_fit", description="多项式拟合：y = a₀ + a₁x + a₂x² + ...")
def polynomial_fit(x: List[float], y: List[float], degree: int = 2) -> Dict[str, Any]:
    """计算多项式拟合参数"""
    try:
        if len(x) != len(y):
            raise ValueError("x和y数据点数量必须相同")

        if len(x) <= degree:
            raise ValueError(f"数据点数量({len(x)})必须大于多项式阶数({degree})")

        x_np = np.array(x, dtype=np.float64)
        y_np = np.array(y, dtype=np.float64)

        coeffs = np.polyfit(x_np, y_np, degree)
        p = np.poly1d(coeffs)

        # 计算拟合优度
        y_pred = p(x_np)
        residuals = y_np - y_pred
        ss_tot = np.sum((y_np - np.mean(y_np)) ** 2)
        ss_res = np.sum(residuals ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # 构建多项式表达式
        terms = []
        for i, coef in enumerate(coeffs):
            power = degree - i
            if power == 0:
                terms.append(f"{coef:.6g}")
            elif power == 1:
                terms.append(f"{coef:.6g}x")
            else:
                terms.append(f"{coef:.6g}x^{power}")
        equation = " + ".join(terms).replace("+ -", "- ")

        return {
            "coefficients": coeffs.tolist(),
            "r_squared": float(r_squared),
            "equation": equation,
            "rmse": float(np.sqrt(np.mean(residuals ** 2)))
        }
    except Exception as e:
        raise ValueError(f"多项式拟合失败: {str(e)}")


@app.tool(name="data_interpolation", description="数据插值")
def data_interpolation(
        x: List[float],
        y: List[float],
        x_new: Union[float, List[float]],
        method: str = 'linear'
) -> Union[float, List[float]]:
    """在给定点上进行数据插值"""
    try:
        if len(x) != len(y):
            raise ValueError("x和y数据点数量必须相同")

        if len(x) < 2:
            raise ValueError("至少需要2个数据点进行插值")

        # 验证插值方法
        valid_methods = ['linear', 'cubic', 'nearest', 'quadratic']
        if method not in valid_methods:
            raise ValueError(f"不支持的插值方法: {method}，可用方法: {', '.join(valid_methods)}")

        x_np = np.array(x, dtype=np.float64)
        y_np = np.array(y, dtype=np.float64)

        # 检查x是否有序
        if not np.all(np.diff(x_np) >= 0):
            # 对数据进行排序
            indices = np.argsort(x_np)
            x_np = x_np[indices]
            y_np = y_np[indices]

        from scipy.interpolate import interp1d

        f = interp1d(x_np, y_np, kind=method, bounds_error=False, fill_value="extrapolate")

        # 处理单个插值点或多个插值点
        if isinstance(x_new, (int, float)):
            return float(f(x_new))
        else:
            x_new_np = np.array(x_new, dtype=np.float64)
            results = f(x_new_np)
            return [float(y) for y in results]
    except Exception as e:
        raise ValueError(f"插值计算失败: {str(e)}")


# ===== 14. 特殊函数计算 =====
@app.tool(name="factorial", description="阶乘：n!")
def factorial(n: int) -> float:
    """计算整数的阶乘"""
    try:
        if not isinstance(n, int):
            raise ValueError("阶乘只适用于整数")
        if n < 0:
            raise ValueError("阶乘不适用于负数")
        if n > 170:
            raise ValueError("数值过大，可能导致溢出")

        return float(math.factorial(n))
    except Exception as e:
        raise ValueError(f"阶乘计算失败: {str(e)}")


@app.tool(name="gamma", description="伽玛函数：Γ(x)")
def gamma_function(x: float) -> float:
    """计算伽玛函数值"""
    try:
        return float(scipy.special.gamma(x))
    except Exception as e:
        raise ValueError(f"伽玛函数计算失败: {str(e)}")


@app.tool(name="beta", description="贝塔函数：B(a,b)")
def beta_function(a: float, b: float) -> float:
    """计算贝塔函数值"""
    try:
        return float(scipy.special.beta(a, b))
    except Exception as e:
        raise ValueError(f"贝塔函数计算失败: {str(e)}")


@app.tool(name="erf", description="误差函数：erf(x)")
def error_function(x: float) -> float:
    """计算误差函数值"""
    try:
        return float(scipy.special.erf(x))
    except Exception as e:
        raise ValueError(f"误差函数计算失败: {str(e)}")


@app.tool(name="bessel", description="贝塞尔函数：J_n(x)")
def bessel_function(n: int, x: float) -> float:
    """计算第一类贝塞尔函数值"""
    try:
        return float(scipy.special.jv(n, x))
    except Exception as e:
        raise ValueError(f"贝塞尔函数计算失败: {str(e)}")


# ===== 15. 复数运算 =====
@app.tool(name="complex_number", description="创建复数：a+bi")
def create_complex(real: float, imag: float) -> Dict[str, float]:
    """创建复数并返回其属性"""
    try:
        z = complex(real, imag)
        return {
            "real": float(z.real),
            "imag": float(z.imag),
            "modulus": float(abs(z)),
            "phase_rad": float(np.angle(z, deg=False)),
            "phase_deg": float(np.angle(z, deg=True))
        }
    except Exception as e:
        raise ValueError(f"复数创建失败: {str(e)}")


@app.tool(name="complex_add", description="复数加法：(a+bi)+(c+di)")
def complex_add(z1: Dict[str, float], z2: Dict[str, float]) -> Dict[str, float]:
    """计算两个复数的和"""
    try:
        c1 = complex(z1["real"], z1["imag"])
        c2 = complex(z2["real"], z2["imag"])
        result = c1 + c2
        return {
            "real": float(result.real),
            "imag": float(result.imag),
            "modulus": float(abs(result)),
            "phase_rad": float(np.angle(result, deg=False))
        }
    except Exception as e:
        raise ValueError(f"复数加法计算失败: {str(e)}")


@app.tool(name="complex_multiply", description="复数乘法：(a+bi)×(c+di)")
def complex_multiply(z1: Dict[str, float], z2: Dict[str, float]) -> Dict[str, float]:
    """计算两个复数的积"""
    try:
        c1 = complex(z1["real"], z1["imag"])
        c2 = complex(z2["real"], z2["imag"])
        result = c1 * c2
        return {
            "real": float(result.real),
            "imag": float(result.imag),
            "modulus": float(abs(result)),
            "phase_rad": float(np.angle(result, deg=False))
        }
    except Exception as e:
        raise ValueError(f"复数乘法计算失败: {str(e)}")


@app.tool(name="complex_divide", description="复数除法：(a+bi)÷(c+di)")
def complex_divide(z1: Dict[str, float], z2: Dict[str, float]) -> Dict[str, float]:
    """计算两个复数的商"""
    try:
        c1 = complex(z1["real"], z1["imag"])
        c2 = complex(z2["real"], z2["imag"])
        if abs(c2) < 1e-15:
            raise ValueError("除数不能为零")
        result = c1 / c2
        return {
            "real": float(result.real),
            "imag": float(result.imag),
            "modulus": float(abs(result)),
            "phase_rad": float(np.angle(result, deg=False))
        }
    except Exception as e:
        raise ValueError(f"复数除法计算失败: {str(e)}")


@app.tool(name="complex_power", description="复数幂：(a+bi)^n")
def complex_power(z: Dict[str, float], n: float) -> Dict[str, float]:
    """计算复数的幂"""
    try:
        c = complex(z["real"], z["imag"])
        result = c ** n
        return {
            "real": float(result.real),
            "imag": float(result.imag),
            "modulus": float(abs(result)),
            "phase_rad": float(np.angle(result, deg=False))
        }
    except Exception as e:
        raise ValueError(f"复数幂计算失败: {str(e)}")


# ===== 16. 概率分布与随机数 =====
@app.tool(name="normal_distribution", description="正态分布计算")
def normal_distribution(params: Dict[str, Any]) -> Dict[str, Any]:
    """计算正态分布的概率密度函数(PDF)、累积分布函数(CDF)或生成随机数"""
    try:
        op = params.get("operation", "pdf")
        mean = params.get("mean", 0.0)
        std = params.get("std", 1.0)

        if std <= 0:
            raise ValueError("标准差必须为正数")

        if op == "pdf":
            x = params.get("x")
            if x is None:
                raise ValueError("计算PDF需要提供x值")
            return {"pdf": float(scipy.stats.norm.pdf(x, loc=mean, scale=std))}

        elif op == "cdf":
            x = params.get("x")
            if x is None:
                raise ValueError("计算CDF需要提供x值")
            return {"cdf": float(scipy.stats.norm.cdf(x, loc=mean, scale=std))}

        elif op == "ppf":  # 百分位点函数（反CDF）
            p = params.get("p")
            if p is None:
                raise ValueError("计算分位数需要提供p值")
            if p < 0 or p > 1:
                raise ValueError("p值必须在[0,1]范围内")
            return {"x": float(scipy.stats.norm.ppf(p, loc=mean, scale=std))}

        elif op == "random":
            size = params.get("size", 1)
            if not isinstance(size, int) or size <= 0:
                raise ValueError("size必须是正整数")
            return {"random": scipy.stats.norm.rvs(loc=mean, scale=std, size=size).tolist()}

        else:
            raise ValueError(f"不支持的操作: {op}")
    except Exception as e:
        raise ValueError(f"正态分布计算失败: {str(e)}")


@app.tool(name="binomial_distribution", description="二项分布计算")
def binomial_distribution(params: Dict[str, Any]) -> Dict[str, Any]:
    """计算二项分布的概率质量函数(PMF)、累积分布函数(CDF)或生成随机数"""
    try:
        op = params.get("operation", "pmf")
        n = params.get("n")
        p = params.get("p")

        if n is None or p is None:
            raise ValueError("必须提供参数n和p")

        if not isinstance(n, int) or n <= 0:
            raise ValueError("n必须是正整数")

        if p < 0 or p > 1:
            raise ValueError("p必须在[0,1]范围内")

        if op == "pmf":
            k = params.get("k")
            if k is None:
                raise ValueError("计算PMF需要提供k值")
            return {"pmf": float(scipy.stats.binom.pmf(k, n, p))}

        elif op == "cdf":
            k = params.get("k")
            if k is None:
                raise ValueError("计算CDF需要提供k值")
            return {"cdf": float(scipy.stats.binom.cdf(k, n, p))}

        elif op == "random":
            size = params.get("size", 1)
            if not isinstance(size, int) or size <= 0:
                raise ValueError("size必须是正整数")
            return {"random": scipy.stats.binom.rvs(n, p, size=size).tolist()}

        else:
            raise ValueError(f"不支持的操作: {op}")
    except Exception as e:
        raise ValueError(f"二项分布计算失败: {str(e)}")


# ===== 17. 数值方程求解 =====
@app.tool(name="solve_equation", description="数值方程求解：f(x)=0")
def solve_equation(
        f: str,
        x0: float,
        bounds: Optional[List[float]] = None,
        tol: float = 1e-10
) -> Dict[str, Any]:
    """数值方法求解方程f(x)=0"""
    try:
        # 创建安全的数学环境
        math_env = evaluator.allowed_names.copy()

        # 清理表达式
        f = f.replace('^', '**')

        # 创建函数
        def func(t: Union[float, np.ndarray]) -> float:
            if isinstance(t, (list, tuple, np.ndarray)):
                t = t[0]  # 处理scipy某些优化函数传入数组的情况
            local_vars = {'x': t, **math_env}
            return eval(f, {'__builtins__': None}, local_vars)

        # 使用合适的求解方法
        if bounds:
            if len(bounds) != 2:
                raise ValueError("bounds必须是[下界,上界]格式")
            if bounds[0] >= bounds[1]:
                raise ValueError("下界必须小于上界")

            # 使用有边界的方法
            from scipy.optimize import brentq
            try:
                # 先检查边界点的函数值是否异号，这是二分法的前提
                f_lower = func(bounds[0])
                f_upper = func(bounds[1])

                if f_lower * f_upper > 0:
                    raise ValueError("区间端点函数值必须异号才能使用二分法")

                root = brentq(func, bounds[0], bounds[1], rtol=tol)
                return {
                    "root": float(root),
                    "function_value": float(func(root)),
                    "success": True,
                    "method": "brentq"
                }
            except ValueError:
                # 如果边界点函数值同号，尝试其他方法
                from scipy.optimize import minimize_scalar
                result = minimize_scalar(
                    lambda x: abs(func(x)),
                    bounds=bounds,
                    method='bounded'
                )

                if abs(func(result.x)) < tol:
                    return {
                        "root": float(result.x),
                        "function_value": float(func(result.x)),
                        "success": True,
                        "method": "minimize_abs"
                    }
                else:
                    raise ValueError(f"在指定区间内未找到解，最小函数值：{func(result.x)}")
        else:
            # 无边界约束，使用牛顿法系列
            from scipy.optimize import root
            result = root(func, x0)

            if result.success:
                return {
                    "root": float(result.x[0]),
                    "function_value": float(func(result.x[0])),
                    "iterations": int(result.nfev),
                    "success": True,
                    "method": "newton"
                }
            else:
                # 如果失败，尝试其他方法
                from scipy.optimize import fsolve
                root_value = fsolve(func, x0)[0]
                is_success = abs(func(root_value)) < tol

                return {
                    "root": float(root_value),
                    "function_value": float(func(root_value)),
                    "success": is_success,
                    "method": "fsolve"
                }
    except Exception as e:
        raise ValueError(f"方程求解失败: {str(e)}")


@app.tool(name="find_roots_polynomial", description="多项式求根")
def find_roots_polynomial(coefficients: List[float]) -> Dict[str, Any]:
    """求多项式的所有根"""
    try:
        if not coefficients:
            raise ValueError("系数列表不能为空")

        # 确保最高次幂系数不为零
        if abs(coefficients[0]) < 1e-15:
            raise ValueError("最高次幂系数不能为零")

        # NumPy的多项式求根，系数顺序从高次到低次
        roots = np.roots(coefficients)

        # 将根分为实根和复根
        real_roots = []
        complex_roots = []

        for root in roots:
            if abs(root.imag) < 1e-10:  # 如果虚部很小，认为是实根
                real_roots.append(float(root.real))
            else:
                complex_roots.append(complex(root.real, root.imag))

        return {
            "real_roots": real_roots,
            "complex_roots": [(r.real, r.imag) for r in complex_roots],
            "all_roots": [(float(r.real), float(r.imag)) for r in roots]
        }
    except Exception as e:
        raise ValueError(f"多项式求根失败: {str(e)}")


# ===== 18. 微分方程求解 =====
@app.tool(name="solve_ode", description="求解常微分方程：dy/dx = f(x,y)")
def solve_ode(
        f: str,
        x_range: List[float],
        y0: float,
        num_points: int = 100
) -> Dict[str, Any]:
    """数值方法求解一阶常微分方程"""
    try:
        if len(x_range) != 2:
            raise ValueError("x_range应为[起点,终点]格式")

        if x_range[0] >= x_range[1]:
            raise ValueError("x范围的起点必须小于终点")

        if num_points < 2:
            raise ValueError("采样点数必须大于1")

        # 创建安全的数学环境
        math_env = evaluator.allowed_names.copy()

        # 清理表达式
        f = f.replace('^', '**')

        # 创建函数
        def func(x: float, y: float) -> float:
            local_vars = {'x': x, 'y': y, **math_env}
            return eval(f, {'__builtins__': None}, local_vars)

        from scipy.integrate import solve_ivp

        # 定义微分方程
        def ode_func(x: float, y: np.ndarray) -> List[float]:
            return [func(x, y[0])]

        # 求解
        sol = solve_ivp(
            ode_func,
            [x_range[0], x_range[1]],
            [y0],
            method='RK45',
            t_eval=np.linspace(x_range[0], x_range[1], num_points)
        )

        return {
            "x": sol.t.tolist(),
            "y": sol.y[0].tolist(),
            "success": bool(sol.success),
            "method": "RK45"
        }
    except Exception as e:
        raise ValueError(f"微分方程求解失败: {str(e)}")


# ===== 19. 优化与极值问题 =====
@app.tool(name="find_minimum", description="函数最小值：min f(x)")
def find_minimum(
        f: str,
        bounds: List[float],
        method: str = 'auto'
) -> Dict[str, Any]:
    """求函数在给定区间上的最小值"""
    try:
        if len(bounds) != 2:
            raise ValueError("bounds应为[下界,上界]格式")

        if bounds[0] >= bounds[1]:
            raise ValueError("下界必须小于上界")

        # 创建安全的数学环境
        math_env = evaluator.allowed_names.copy()

        # 清理表达式
        f = f.replace('^', '**')

        # 创建函数
        def func(x: Union[float, np.ndarray]) -> float:
            if isinstance(x, (list, tuple, np.ndarray)) and len(x) == 1:
                x = x[0]  # 处理优化函数传入数组的情况
            local_vars = {'x': x, **math_env}
            return eval(f, {'__builtins__': None}, local_vars)

        # 确定最佳方法
        if method == 'auto':
            method = 'bounded'  # 默认使用有界优化

        from scipy.optimize import minimize_scalar

        # 执行最小化
        result = minimize_scalar(func, bounds=bounds, method=method)

        # 如果成功，返回结果
        if result.success:
            return {
                "minimum_point": float(result.x),
                "minimum_value": float(result.fun),
                "success": bool(result.success),
                "method": method
            }
        else:
            # 如果失败，尝试其他方法
            # 通过网格搜索找到一个更好的初始点
            grid_points = np.linspace(bounds[0], bounds[1], 10)
            grid_values = [func(x) for x in grid_points]
            best_idx = np.argmin(grid_values)

            from scipy.optimize import minimize
            backup_result = minimize(
                func,
                [grid_points[best_idx]],
                bounds=[bounds],
                method='L-BFGS-B'
            )

            return {
                "minimum_point": float(backup_result.x[0]),
                "minimum_value": float(backup_result.fun),
                "success": bool(backup_result.success),
                "method": "L-BFGS-B (fallback)"
            }
    except Exception as e:
        raise ValueError(f"最小值查找失败: {str(e)}")


@app.tool(name="find_maximum", description="函数最大值：max f(x)")
def find_maximum(
        f: str,
        bounds: List[float],
        method: str = 'auto'
) -> Dict[str, Any]:
    """求函数在给定区间上的最大值"""
    try:
        if len(bounds) != 2:
            raise ValueError("bounds应为[下界,上界]格式")

        if bounds[0] >= bounds[1]:
            raise ValueError("下界必须小于上界")

        # 创建安全的数学环境
        math_env = evaluator.allowed_names.copy()

        # 清理表达式
        f = f.replace('^', '**')

        # 创建负函数（最大化问题转换为最小化问题）
        def negative_func(x: Union[float, np.ndarray]) -> float:
            if isinstance(x, (list, tuple, np.ndarray)) and len(x) == 1:
                x = x[0]
            local_vars = {'x': x, **math_env}
            return -eval(f, {'__builtins__': None}, local_vars)

        # 确定最佳方法
        if method == 'auto':
            method = 'bounded'  # 默认使用有界优化

        from scipy.optimize import minimize_scalar

        # 执行最大化
        result = minimize_scalar(negative_func, bounds=bounds, method=method)

        # 如果成功，返回结果
        if result.success:
            return {
                "maximum_point": float(result.x),
                "maximum_value": float(-result.fun),
                "success": bool(result.success),
                "method": method
            }
        else:
            # 如果失败，尝试其他方法
            # 通过网格搜索找到一个更好的初始点
            grid_points = np.linspace(bounds[0], bounds[1], 10)
            grid_values = [negative_func(x) for x in grid_points]
            best_idx = np.argmin(grid_values)

            from scipy.optimize import minimize
            backup_result = minimize(
                negative_func,
                [grid_points[best_idx]],
                bounds=[bounds],
                method='L-BFGS-B'
            )

            return {
                "maximum_point": float(backup_result.x[0]),
                "maximum_value": float(-backup_result.fun),
                "success": bool(backup_result.success),
                "method": "L-BFGS-B (fallback)"
            }
    except Exception as e:
        raise ValueError(f"最大值查找失败: {str(e)}")


# ===== 20. 综合型表达式计算器 =====
@app.tool(name="calculator", description="全能数学表达式计算器")
def calculator(expr: str, variables: Optional[Dict[str, float]] = None) -> Any:
    """安全计算任意数学表达式"""
    try:
        return evaluator.eval(expr, variables)
    except Exception as e:
        raise ValueError(f"表达式计算失败: {str(e)}")


if __name__ == "__main__":
    # 以标准 I/O 方式运行 MCP 服务器
    app.run(transport='stdio')
