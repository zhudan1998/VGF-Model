import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import matplotlib
import time
from matplotlib import font_manager

# 十二、辅助类定义 - 移到前面，解决NameError
class SimpleClassicsProcessor:
    def __init__(self):
        self.classics_library = {
            '诗经': ['关雎', '蒹葭', '桃夭', '鹿鸣'],
            '论语': ['学而篇', '为政篇', '八佾篇', '里仁篇'],
            '大学': ['三纲领', '八条目'],
            '中庸': ['中庸之道', '诚明之义'],
            '孟子': ['性善论', '仁义礼智']
        }
      
    def create_guanju_input(self):
        """返回《关雎》文本"""
        return [
            "关关雎鸠，在河之洲。",
            "窈窕淑女，君子好逑。", 
            "参差荇菜，左右流之。",
            "窈窕淑女，寤寐求之。",
            "求之不得，寤寐思服。",
            "悠哉悠哉，辗转反侧。",
            "参差荇菜，左右采之。",
            "窈窕淑女，琴瑟友之。",
            "参差荇菜，左右芼之。",
            "窈窕淑女，钟鼓乐之。"
        ]
    
    def get_classic_chapters(self, classic_name):
        """获取指定经典的篇章列表"""
        return self.classics_library.get(classic_name, [])

# 设置中文字体支持
matplotlib.rcParams["font.family"] = ["WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS", "sans-serif"]
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 确保matplotlib能找到可用字体
def setup_fonts():
    preferred_fonts = ["Arial Unicode MS", "SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    
    # 尝试获取系统中安装的字体列表
    font_names = set()
    for font_path in font_manager.findSystemFonts():
        try:
            font = font_manager.get_font(font_path)
            font_names.add(font.family_name)
        except:
            continue
    
    # 选择第一个可用的首选字体
    for font in preferred_fonts:
        if font in font_names:
            matplotlib.rcParams["font.family"] = [font]
            return
    matplotlib.rcParams["font.family"] = ["sans-serif"]

# 执行字体设置
setup_fonts()

# 一、参数配置
class Params:
    def __init__(self):
        self.STATE_DIM = 5  # 阿赖耶识种子状态维度
        self.INPUT_DIM = 5  # 输入维度
        self.MEMORY_CAPACITY = 5  # 短期记忆容量
        self.DECAY_RATE = 0.9  # 记忆衰减率
        self.LEARNING_RATE = 0.15  # 学习率
        self.CONTEXT_WEIGHT = 0.2  # 上下文权重
        self.SEEKING_THRESHOLD = 0.08  # 寻求心阈值
        self.INTEGRATION_STRENGTH_SCALE = 1.5  # 整合强度缩放因子

params = Params()

# 二、基础组件
class ModalityType(Enum):
    """Modality Type Enum"""
    AUDITORY = "Auditory"
    VISUAL = "Visual"
    CONCEPTUAL = "Conceptual"
    TACTILE = "Tactile"
    OLFACTORY = "Olfactory"

class MultiModalInput:
    """多模态输入系统"""
    
    def __init__(self, modalities, sequence_duration=3):
        self.modalities = modalities  # 模态数据
        self.time_indices = {modality: 0 for modality in modalities}  # 每个模态的时间索引
        self.fusion_weights = {modality: 1.0/len(modalities) for modality in modalities}  # 初始融合权重
        self.context = np.zeros(params.INPUT_DIM)  # 上下文信息
        self.sequence_duration = sequence_duration  # 每个输入单元持续的时间步数
        self.current_step_in_sequence = {modality: 0 for modality in modalities}  # 每个模态在当前序列项中的步数
        self.sequence_length = len(next(iter(modalities.values()))) if modalities else 0
        
    def get_input(self, t):
        """获取融合的多模态输入"""
        inputs = []
        
        # 更新每个模态的索引
        for modality in self.modalities:
            if self.current_step_in_sequence[modality] >= self.sequence_duration:
                self.time_indices[modality] = (self.time_indices[modality] + 1) % len(self.modalities[modality])
                self.current_step_in_sequence[modality] = 0
            else:
                self.current_step_in_sequence[modality] += 1
        
        # 收集所有模态的当前输入
        for modality, data in self.modalities.items():
            idx = self.time_indices[modality]
            item = data[idx]
            
            # 处理不同类型的输入
            if isinstance(item, str):  # 文本输入
                vec = np.array([(hash(c) % 20 - 10) / 10.0 for c in item[:params.INPUT_DIM]])
                if len(vec) < params.INPUT_DIM:
                    vec = np.pad(vec, (0, params.INPUT_DIM - len(vec)), mode='constant')
                inputs.append(vec * self.fusion_weights[modality])
            elif isinstance(item, np.ndarray) and len(item) == params.INPUT_DIM:  # 已经是向量
                inputs.append(item * self.fusion_weights[modality])
            else:  # 其他类型
                vec = np.ones(params.INPUT_DIM) * ((hash(str(item)) % 20 - 10) / 10.0)
                inputs.append(vec * self.fusion_weights[modality])
        
        # 融合输入并更新上下文
        fused_input = np.mean(inputs, axis=0) if inputs else np.zeros(params.INPUT_DIM)
        self.context = 0.6 * self.context + 0.4 * fused_input
        
        return fused_input
    
    def get_context(self):
        return self.context
    
    def get_current_item(self, modality):
        if modality in self.modalities:
            idx = self.time_indices[modality]
            return self.modalities[modality][idx]
        return None

    def optimize_modality_weights(self):
        """优化模态权重，增强概念模态的影响"""
        # 只更新已存在的模态权重，修复键不匹配问题
        target_weights = {
            ModalityType.AUDITORY: 0.3,    # 韵律美感
            ModalityType.VISUAL: 0.3,      # 意象构建  
            ModalityType.CONCEPTUAL: 0.4    # 道德内涵（增强）
        }
        
        # 只更新存在的模态
        for modality in self.fusion_weights:
            if modality in target_weights:
                self.fusion_weights[modality] = target_weights[modality]

class RobustShortTermMemory:
    def __init__(self, capacity=5, base_strength=0.1):
        self._capacity = capacity  # 使用私有变量存储容量
        self.base_strength = base_strength
        self.memory_slots = [None] * capacity
        self.strengths = [0.0] * capacity
        self.time_stamps = [0] * capacity
    
    @property
    def capacity(self):
        return self._capacity
    
    @capacity.setter
    def capacity(self, new_capacity):
        """设置新容量并调整相关列表长度"""
        if new_capacity < 1:
            new_capacity = 1  # 确保容量至少为1
            
        old_capacity = self._capacity
        self._capacity = new_capacity
        
        # 如果容量增加，扩展列表
        if new_capacity > old_capacity:
            self.memory_slots.extend([None] * (new_capacity - old_capacity))
            self.strengths.extend([0.0] * (new_capacity - old_capacity))
            self.time_stamps.extend([0] * (new_capacity - old_capacity))
        # 如果容量减少，截断列表
        elif new_capacity < old_capacity:
            self.memory_slots = self.memory_slots[:new_capacity]
            self.strengths = self.strengths[:new_capacity]
            self.time_stamps = self.time_stamps[:new_capacity]
        
    def add_memory(self, content, current_time, importance=1.0):
        if content is None or np.linalg.norm(content) < 0.001:
            content = np.random.randn(params.STATE_DIM) * 0.01
            
        min_strength_idx = np.argmin(self.strengths)
        self.memory_slots[min_strength_idx] = content.copy()
        self.strengths[min_strength_idx] = self.base_strength * importance
        self.time_stamps[min_strength_idx] = current_time
        
        return True
        
    def retrieve_memory(self, current_time):
        total_strength = sum(self.strengths)
        if total_strength < 0.001:
            return np.random.randn(params.STATE_DIM) * 0.01
            
        result = np.zeros(params.STATE_DIM)
        for i in range(self.capacity):
            if self.memory_slots[i] is not None:
                time_diff = current_time - self.time_stamps[i]
                decay_factor = max(0, 1.0 - time_diff * 0.01)
                weight = self.strengths[i] * decay_factor
                result += self.memory_slots[i] * weight
                
        return result / (total_strength + 1e-8)
        
    def get_memory_strength(self):
        return sum(self.strengths)

# 三、规范场论实现
class YangMillsField:
    """杨-米尔斯规范场，模拟心识活动的规范场论模型"""
    
    def __init__(self):
        self.dim = params.STATE_DIM
        self.A_mu = np.zeros((self.dim, self.dim))  # 规范势
        self.F_mu_nu = np.zeros((self.dim, self.dim))  # 场强张量
        self.current = np.zeros((self.dim, self.dim))  # 电流
        self.history = {
            'A_mu': [],
            'F_mu_nu': [],
            'current': []
        }
        self.coupling = 0.4  # 耦合常数
        
    def update_current(self, s, mental_factors):
        kleshas = mental_factors.current_klesha_factors
        virtuous = mental_factors.current_virtuous
        
        klesha_sum = np.sum(kleshas)
        virtuous_sum = np.sum(virtuous)
        
        current = np.diag(np.full(self.dim, klesha_sum - virtuous_sum))
        for i in range(self.dim):
            for j in range(self.dim):
                if i != j:
                    current[i, j] = 0.15 * s[i] * s[j]
                    
        self.current = current
        self.history['current'].append(current.copy())
        if len(self.history['current']) > 100:
            self.history['current'].pop(0)
    
    def yang_mills_equation(self):
        # 修复梯度计算处理，取梯度的迹作为时间导数
        grad_A = np.gradient(self.A_mu)
        dA_dt = np.trace(grad_A[0])  # 取第一个维度的梯度迹作为时间导数
        
        # 计算对易子
        commutator = np.zeros_like(self.A_mu)
        for g in grad_A:
            commutator += np.dot(self.A_mu, g) - np.dot(g, self.A_mu)
        
        # 计算场强张量
        self.F_mu_nu = dA_dt * np.eye(self.dim) - self.coupling * commutator
        
        # 更新规范势
        self.A_mu += 0.08 * (self.current - np.trace(self.F_mu_nu) * np.eye(self.dim))
        
        self.history['A_mu'].append(self.A_mu.copy())
        self.history['F_mu_nu'].append(self.F_mu_nu.copy())
        
        if len(self.history['A_mu']) > 100:
            self.history['A_mu'].pop(0)
        if len(self.history['F_mu_nu']) > 100:
            self.history['F_mu_nu'].pop(0)
    
    def apply_to_state(self, s):
        theta = np.trace(self.A_mu) * 0.1
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        n = len(s)
        full_rot = np.eye(n)
        min_dim = min(2, n)
        full_rot[:min_dim, :min_dim] = rot_matrix[:min_dim, :min_dim]
        
        s_transformed = np.dot(full_rot, s)
        s_transformed += 0.07 * np.dot(self.F_mu_nu, s)
        
        return s_transformed

# 四、心所系统
class CompleteMentalFactors:
    """完整的心所系统，包括遍行、别境、善、烦恼心所"""
    
    def __init__(self):
        self.universal_weights = {
            'contact': 0.25,
            'attention': 0.25,
            'feeling': 0.2,
            'perception': 0.15,
            'volition': 0.15
        }
        
        self.particular_weights = {
            'desire': 0.2,
            'resolution': 0.2,
            'mindfulness': 0.25,
            'concentration': 0.15,
            'wisdom': 0.2
        }
        
        self.virtue_weights = {
            'non_greed': 0.2,
            'non_hatred': 0.2,
            'non_ignorance': 0.2,
            'diligence': 0.1,
            'tranquility': 0.1,
            'carefulness': 0.1,
            'equanimity': 0.1
        }
        
        self.klesha_weights = {
            'greed': 0.2,
            'hatred': 0.2,
            'ignorance': 0.2,
            'conceit': 0.15,
            'doubt': 0.15,
            'wrong_views': 0.1
        }
        
        self.secondary_factors = {
            'anger': 0.0,
            'envy': 0.0,
            'jealousy': 0.0,
            'arrogance': 0.0,
            'deceit': 0.0,
            'hypocrisy': 0.0,
            'sloth': 0.0,
            'torpor': 0.0,
            'restlessness': 0.0,
            'remorse': 0.0
        }
        
        self.universal_history = []
        self.particular_history = []
        self.virtuous_history = []
        self.klesha_history = []
        
        self.current_virtuous = []
        self.current_klesha_factors = []
        self.current_feeling = 0
    
    def universal_factors(self, s, current_input, memory, current_time):
        contact = self.universal_weights['contact'] * np.dot(s, current_input)
        attention = self.universal_weights['attention'] * np.linalg.norm(s)
        
        memory_content = memory.retrieve_memory(current_time)
        feeling = self.universal_weights['feeling'] * np.dot(current_input, memory_content) * 1.2
        self.current_feeling = feeling
        
        perception = self.universal_weights['perception'] * np.tanh(current_input + memory_content)
        volition = self.universal_weights['volition'] * np.tanh(s)
        
        universal = [contact, attention, feeling, perception, volition]
        self.current_universal = universal
        
        self.universal_history.append(self.current_universal)
        if len(self.universal_history) > 100:
            self.universal_history.pop(0)
            
        return universal
    
    def particular_factors(self, s, perception, memory, current_time):
        desire = self.particular_weights['desire'] * np.max(s)
        resolution = self.particular_weights['resolution'] * np.linalg.norm(perception)
        
        memory_content = memory.retrieve_memory(current_time)
        memory_strength = memory.get_memory_strength()
        mindfulness = self.particular_weights['mindfulness'] * (np.linalg.norm(memory_content) + memory_strength)
        
        concentration = self.particular_weights['concentration'] * (1 - np.std(s))
        wisdom = self.particular_weights['wisdom'] * np.linalg.norm(np.tanh(perception))
        
        particular = [desire, resolution, mindfulness, concentration, wisdom]
        self.current_particular = particular
        
        self.particular_history.append(self.current_particular)
        if len(self.particular_history) > 100:
            self.particular_history.pop(0)
            
        return particular
    
    def virtuous_factors(self, wisdom, perception):
        non_greed = self.virtue_weights['non_greed'] * (1 - np.max(perception))
        non_hatred = self.virtue_weights['non_hatred'] * (1 + np.min(perception))
        non_ignorance = self.virtue_weights['non_ignorance'] * np.linalg.norm(wisdom)
        diligence = self.virtue_weights['diligence'] * np.linalg.norm(wisdom)
        tranquility = self.virtue_weights['tranquility'] * (1 - np.var(perception))
        carefulness = self.virtue_weights['carefulness'] * (1 - np.std(perception))
        equanimity = self.virtue_weights['equanimity'] * (1 - np.abs(np.mean(perception)))
        
        faith = 0.1 * (non_greed + non_hatred + non_ignorance)
        self_respect = 0.1 * (diligence + carefulness)
        respect_for_others = 0.1 * (non_hatred + equanimity)
        non_harming = self.virtue_weights['non_greed'] * respect_for_others
        non_attachment = 0.1 * (equanimity + non_greed)
        
        virtuous = [faith, self_respect, respect_for_others, non_attachment, 
                   non_hatred, non_ignorance, diligence, tranquility, 
                   carefulness, equanimity, non_harming]
        
        self.current_virtuous = virtuous
        self.virtuous_history.append(self.current_virtuous)
        if len(self.virtuous_history) > 100:
            self.virtuous_history.pop(0)
            
        return virtuous
    
    def klesha_factors(self, s, perception):
        greed = self.klesha_weights['greed'] * max(0, self.current_feeling)
        hatred = self.klesha_weights['hatred'] * max(0, -self.current_feeling)
        ignorance = self.klesha_weights['ignorance'] * (1 - np.linalg.norm(perception))
        conceit = self.klesha_weights['conceit'] * np.linalg.norm(s)
        doubt = self.klesha_weights['doubt'] * (1 - np.linalg.norm(perception))
        wrong_views = self.klesha_weights['wrong_views'] * ignorance
        
        kleshas = [greed, hatred, ignorance, conceit, doubt, wrong_views]
        self.current_klesha_factors = kleshas
        
        self.current_klesha = kleshas
        self.klesha_history.append(self.current_klesha)
        if len(self.klesha_history) > 100:
            self.klesha_history.pop(0)
            
        return kleshas
    
    def update_secondary_factors(self, kleshas):
        greed, hatred, ignorance, conceit, doubt, wrong_views = kleshas
        
        self.secondary_factors['anger'] = 0.7 * hatred
        self.secondary_factors['envy'] = 0.5 * greed
        self.secondary_factors['jealousy'] = 0.6 * (greed + hatred)
        self.secondary_factors['arrogance'] = 0.8 * conceit
        self.secondary_factors['deceit'] = 0.4 * wrong_views
        self.secondary_factors['hypocrisy'] = 0.3 * wrong_views
        self.secondary_factors['sloth'] = 0.5 * ignorance
        self.secondary_factors['torpor'] = 0.4 * ignorance
        self.secondary_factors['restlessness'] = 0.6 * (hatred + greed)
        self.secondary_factors['remorse'] = 0.5 * doubt

# 五、结构群与心所的精细映射
class ManasGaugeMapping:
    """末那识四烦恼与结构群的精细映射"""
    
    def __init__(self):
        # SU(2)生成元 - Pauli矩阵
        self.su2_generators = {
            'ignorance': np.array([[0, 1], [1, 0]]),   # σ_x - 我痴
            'view': np.array([[0, -1], [1, 0]]),      # σ_y的实数近似 - 我见
            'pride': np.array([[1, 0], [0, -1]]),      # σ_z - 我慢
        }
        
        # U(1)相位 - 我爱
        self.u1_phase = 0.0
        
        # 心所与烦恼的映射权重
        self.factor_mapping = self.initialize_factor_mapping()
    
    def initialize_factor_mapping(self):
        mapping = {
            # 遍行心所映射
            'contact': {'ignorance': 0.3, 'view': 0.2, 'pride': 0.1, 'love': 0.4},
            'attention': {'ignorance': 0.1, 'view': 0.4, 'pride': 0.3, 'love': 0.2},
            'feeling': {'ignorance': 0.2, 'view': 0.3, 'pride': 0.2, 'love': 0.3},
            'perception': {'ignorance': 0.4, 'view': 0.3, 'pride': 0.1, 'love': 0.2},
            'volition': {'ignorance': 0.1, 'view': 0.2, 'pride': 0.4, 'love': 0.3},
            
            # 别境心所映射
            'desire': {'love': 0.8, 'view': 0.2},
            'resolution': {'view': 0.6, 'ignorance': 0.4},
            'mindfulness': {'view': 0.5, 'love': 0.5},
            'concentration': {'pride': 0.7, 'view': 0.3},
            'wisdom': {'ignorance': -0.9, 'view': 0.1},  # 负权重表示对抗
        }
        return mapping
    
    def apply_manas_effect(self, mental_factor, factor_type, intensity):
        if factor_type not in self.factor_mapping:
            return mental_factor
        
        mapping_weights = self.factor_mapping[factor_type]
        if not isinstance(mental_factor, np.ndarray):
            mental_factor = np.array([mental_factor])
        
        transformed_factor = mental_factor.copy()
        
        for affliction, weight in mapping_weights.items():
            if weight == 0:
                continue
                
            if affliction in ['ignorance', 'view', 'pride']:
                generator = self.su2_generators[affliction]
                n = len(transformed_factor)
                if n == 2:
                    effect = np.dot(generator, transformed_factor)
                else:
                    gen_extended = np.zeros((n, n))
                    min_dim = min(2, n)
                    gen_extended[:min_dim, :min_dim] = generator[:min_dim, :min_dim]
                    effect = np.dot(gen_extended, transformed_factor)
                    
            else:  # love - U(1)作用
                phase = weight * intensity * np.pi
                effect = transformed_factor * np.cos(phase)
                if n > 1:
                    orthogonal_component = transformed_factor[1:]
                    orthogonal_component = np.append(orthogonal_component, transformed_factor[0])
                    effect += orthogonal_component * np.sin(phase)
                
            transformed_factor += weight * effect
            
        if len(transformed_factor) == 1:
            return transformed_factor[0]
        return transformed_factor

# 六、改进的五心状态机
# 修复：定义基础类FiveHeartsStateMachine
class FiveHeartsStateMachine:
    def __init__(self):
        self.current_state = 0  # 0:空闲, 1:率尔心, 2:寻求心, 3:决定心, 4:染净心, 5:等流心
        self.prev_consciousness = 0
        self.time_in_state = 0
        self.transition_history = []
        self.last_input_change_time = 0
        self.input_change_detected = False
        
    def update(self, s, current_input, heart_functions, t, integration_strength, input_changed=False):
        # 默认实现，可在子类中重写
        return self.current_state, 0.0

# 修复：修正继承和缩进问题
class ImprovedFiveHeartsStateMachine(FiveHeartsStateMachine):
    def __init__(self):
        super().__init__()  # 调用父类构造函数
        # 可以在这里添加额外的初始化代码
        
    def update(self, s, current_input, heart_functions, t, integration_strength, input_changed=False):
        # 确保heart_functions至少有5个元素
        if len(heart_functions) >= 5:
            T, A, V, P, C = heart_functions[:5]
        else:
            # 提供默认值防止索引错误
            T, A, V, P, C = [0.0]*5
            
        input_strength = np.linalg.norm(current_input)
        consciousness = np.linalg.norm(P)
        dynamic_threshold = params.SEEKING_THRESHOLD * (1 - 0.4 * integration_strength)
        
        # 检测输入变化
        if input_changed:
            self.input_change_detected = True
            self.last_input_change_time = t
        
        prev_state = self.current_state
        
        if self.current_state == 0:  # 空闲
            if input_strength > 0.08 or self.input_change_detected:
                self.current_state = 1
                self.time_in_state = 0
                self.input_change_detected = False
                
        elif self.current_state == 1:  # 率尔心
            if self.time_in_state > max(1, int(3 * (1 - input_strength))):
                self.current_state = 2
                self.time_in_state = 0
                self.prev_consciousness = consciousness
                
        elif self.current_state == 2:  # 寻求心
            consciousness_change = abs(consciousness - self.prev_consciousness)
            if (consciousness_change < dynamic_threshold and self.time_in_state > 2) or self.time_in_state > 8:
                self.current_state = 3
                self.time_in_state = 0
                
        elif self.current_state == 3:  # 决定心
            min_time = max(1, int(2 + 1 * (1 - integration_strength)))
            if self.time_in_state > min_time:
                self.current_state = 4
                self.time_in_state = 0
        
        # 修复：修正方法缩进和内部代码块缩进
        elif self.current_state == 4:  # 染净心
            # 调用增强的净化阶段处理
            if not self.enhanced_purification_phase(integration_strength, heart_functions[5]):  # heart_functions[5]是mindfulness
                self.current_state = 5
                self.time_in_state = 0
                
        elif self.current_state == 5:  # 等流心
            # 改进退出条件：考虑整合强度和输入变化的综合影响
            time_since_input_change = t - self.last_input_change_time
            stability_factor = 1.0 - integration_strength  # 整合强度越高越稳定
            
            if (input_strength < 0.08 and integration_strength < 0.3) or \
               (self.time_in_state > 8 * stability_factor):
                self.current_state = 0
                self.time_in_state = 0
        
        # 记录状态转移
        if self.current_state != prev_state:
            self.transition_history.append((t, prev_state, self.current_state))
            if len(self.transition_history) > 50:
                self.transition_history.pop(0)
            
        self.prev_consciousness = consciousness
        self.time_in_state += 1
        return self.current_state, np.linalg.norm(V)
    
    # 修复：修正方法缩进
    def enhanced_purification_phase(self, integration_strength, mindfulness):
        """增强染净心处理深度"""
        # 道德内化需要时间和深度
        purification_depth = integration_strength * mindfulness
        min_purification_time = max(3, int(6 * purification_depth))
        
        if self.time_in_state < min_purification_time:
            return True  # 保持染净心状态
        return False
    
    def enhanced_state_transition(self, new_state_candidate):
        # 防止过于频繁的状态跳转
        min_state_duration = {
            0: 2,  # 空闲至少2步
            1: 2,  # 率尔心至少2步  
            2: 3,  # 寻求心至少3步
            3: 2,  # 决定心至少2步
            4: 2,  # 染净心至少2步
            5: 4   # 等流心至少4步
        }
        
        if self.time_in_state < min_state_duration[self.current_state]:
            return self.current_state  # 保持当前状态
        
        return new_state_candidate
    
    # 修复：修正方法缩进和格式
    def improved_state_retention(self):
        """增强状态保持机制"""
        state_minimum_duration = {
            0: 3,   # 空闲至少3步
            1: 3,   # 率尔心至少3步  
            2: 4,   # 寻求心至少4步
            3: 3,   # 决定心至少3步
            4: 4,   # 染净心至少4步（道德内化需要时间）
            5: 5    # 等流心至少5步
        }
        return state_minimum_duration
    
    # 修复：修正方法缩进和返回值
    def final_state_optimization(self):
        """最终状态稳定性优化"""
        state_cooldown = {
            0: 2,  # 空闲冷却
            1: 2,  # 率尔心冷却  
            2: 3,  # 寻求心冷却
            3: 2,  # 决定心冷却
            4: 3,  # 染净心冷却（重要状态）
            5: 3   # 等流心冷却
        }
        # 防止状态跳转过于频繁
        if self.time_in_state < state_cooldown.get(self.current_state, 1):
            return self.current_state
        return None

# 七、核心认知功能
# 修复：函数名和实现逻辑
def dynamic_conceptual_integration(current_input, memory_content, time_step, prev_integration=0):
    if np.linalg.norm(current_input) < 0.001:
        current_input = np.random.randn(params.STATE_DIM) * 0.01
        
    if np.linalg.norm(memory_content) < 0.001:
        memory_content = np.random.randn(params.STATE_DIM) * 0.01
    
    similarity = np.dot(current_input, memory_content) / (
        np.linalg.norm(current_input) * np.linalg.norm(memory_content) + 1e-8)
    
    time_factor = 0.5 + 0.5 * np.sin(time_step * 0.1)
    
    # 修复：调整计算顺序，确保变量先定义后使用
    alpha = 0.3 + 0.4 * np.tanh(5 * similarity)
    integrated_concept = alpha * current_input + (1 - alpha) * memory_content
    integrated_concept = np.tanh(integrated_concept)
    
    raw_integration = np.tanh(5 * similarity) * time_factor
    # 添加平滑滤波
    smoothing_factor = 0.7
    smoothed_integration = (smoothing_factor * prev_integration + 
                           (1 - smoothing_factor) * raw_integration)
    
    return integrated_concept, smoothed_integration

def feedback_mechanism(s, current_state, integration_strength, memory):
    if integration_strength < 0.3 and current_state in [2, 3]:
        new_capacity = min(7, memory.capacity + 1)
        memory.capacity = new_capacity
    elif integration_strength > 0.6 and memory.capacity > 3:
        memory.capacity = max(3, memory.capacity - 1)
    
    learning_rate = 0.15
    if current_state == 4:
        learning_rate *= 1.8
    elif current_state == 3:
        learning_rate *= 1.3
    elif current_state == 0:
        learning_rate *= 0.5
    return learning_rate

def imprint_effect(s, feeling_value, learning_rate):
    return s * (1 + learning_rate * feeling_value * 1.5)

def alaya_evolution(s, t, current_input, heart_functions, learning_rate):
    # 确保有足够的元素，防止索引错误
    if len(heart_functions) >= 3:
        T, A, V = heart_functions[:3]
    else:
        T, A, V = 0.0, 0.0, 0.0
        
    combined = (T + A + V) / 3
    return learning_rate * (0.15 * (current_input - s) + 0.08 * combined)

# 修复：移除self参数，使其成为独立函数
def balanced_memory_integration(memory_strength, integration_strength):
    """平衡记忆与整合的关系"""
    # 当整合强度高时，应增强记忆
    if integration_strength > 0.7:
        memory_boost = 0.3 * integration_strength
        return min(1.0, memory_strength + memory_boost)
    return memory_strength

# 修复：移除self参数，修正缩进
def enhanced_memory_integration_sync(integration_strength, memory_strength):
    """增强记忆与整合的同步性"""
    # 当整合强度高时，同步提升记忆强度
    sync_threshold = 0.7
    if integration_strength > sync_threshold:
        memory_boost = 0.4 * (integration_strength - sync_threshold)
        return min(0.8, memory_strength + memory_boost)
    return memory_strength

# 修复：移除self参数，修正缩进
def improved_initial_response(input_strength, total_time):
    """改进初始响应机制"""
    if total_time < 10:  # 前10个时间步
        initial_sensitivity = 1.5  # 提高初始敏感性
        return input_strength * initial_sensitivity > 0.05
    return input_strength > 0.08 

# 八、增强的可视化分析系统
class AdvancedVisualization:
    """Enhanced visualization analysis system"""
    
    def __init__(self, export_format='pdf', dpi=300):
        self.fig = None
        self.color_map = plt.cm.viridis
        self.export_format = export_format  # 'pdf' or 'tiff'
        self.dpi = dpi  # resolution for exports
        
    def create_comprehensive_dashboard(self, results, yang_mills_field, mental_factors):
        """Create comprehensive dashboard"""
        plt.figure(figsize=(20, 20))
        
        # 1. Five hearts state and multimodal input
        plt.subplot(3, 2, 1)
        self.plot_state_and_modality(results)
        
        # 2. Gauge field dynamics
        plt.subplot(3, 2, 2)
        self.plot_gauge_field_dynamics(yang_mills_field)
        
        # 3. Mental factors activity
        plt.subplot(3, 2, 3)
        self.plot_mental_factors_activity(mental_factors)
        
        # 4. Seed evolution trajectory
        ax4 = plt.subplot(3, 2, 4, projection='3d')
        self.plot_seed_evolution(results, ax4)
        
        # 5. Manas kleshas intensity
        plt.subplot(3, 2, 5)
        self.plot_manas_kleshas(mental_factors)
        
        # 6. Consciousness phase portrait
        plt.subplot(3, 2, 6)
        self.plot_consciousness_phase_portrait(results)
        
        plt.tight_layout()
        plt.show()
        
        # Additional state vector parallel coordinates plot
        self.plot_state_parallel_coordinates(results)
        
        # Memory strength and integration strength plot
        self.plot_memory_and_integration(results)
        
        # New: Detailed state transitions analysis plot
        self.plot_detailed_state_transitions(results)
        
    def create_separate_charts(self, results, yang_mills_field, mental_factors, export_path='.'):
        """Create and export each chart separately"""
        
        # Determine file extension based on export format
        if self.export_format == 'excel':
            ext = 'xlsx'
        else:
            ext = self.export_format
        
        # 1. Five hearts state and multimodal input
        self.fig = plt.figure(figsize=(12, 8))
        self.plot_state_and_modality(results)
        self.export_chart(f"{export_path}/five_hearts_state.{ext}")
        
        # 2. Gauge field dynamics
        self.fig = plt.figure(figsize=(12, 8))
        self.plot_gauge_field_dynamics(yang_mills_field)
        self.export_chart(f"{export_path}/gauge_field_dynamics.{ext}")
        
        # 3. Mental factors activity
        self.fig = plt.figure(figsize=(12, 8))
        self.plot_mental_factors_activity(mental_factors)
        self.export_chart(f"{export_path}/mental_factors_activity.{ext}")
        
        # 4. Seed evolution trajectory
        self.fig = plt.figure(figsize=(12, 8))
        ax = self.fig.add_subplot(111, projection='3d')
        self.plot_seed_evolution(results, ax)
        self.export_chart(f"{export_path}/seed_evolution.{ext}")
        
        # 5. Manas kleshas intensity
        self.fig = plt.figure(figsize=(12, 8))
        self.plot_manas_kleshas(mental_factors)
        self.export_chart(f"{export_path}/manas_kleshas.{ext}")
        
        # 6. Consciousness phase portrait
        self.fig = plt.figure(figsize=(12, 8))
        self.plot_consciousness_phase_portrait(results)
        self.export_chart(f"{export_path}/consciousness_phase.{ext}")
        
        # 7. State vector parallel coordinates
        self.fig = plt.figure(figsize=(12, 8))
        self.plot_state_parallel_coordinates(results, show=False)
        self.export_chart(f"{export_path}/state_parallel_coordinates.{ext}")
        
        # 8. Memory and integration strength
        self.fig = plt.figure(figsize=(12, 8))
        self.plot_memory_and_integration(results, show=False)
        self.export_chart(f"{export_path}/memory_integration.{ext}")
        
        # 9. Detailed state transitions
        self.fig = plt.figure(figsize=(15, 10))
        self.plot_detailed_state_transitions(results, show=False)
        self.export_chart(f"{export_path}/state_transitions.{ext}")
        
    def export_chart(self, file_path):
        """Export the current figure to the specified file path"""
        print(f"Exporting chart to: {file_path}")
        try:
            if self.export_format == 'tiff':
                plt.savefig(file_path, dpi=self.dpi, format='tiff', bbox_inches='tight')
            elif self.export_format == 'excel':
                # For Excel export, we need to use a different approach
                # First, save as SVG which is a vector format
                svg_path = file_path.replace('.xlsx', '.svg')
                plt.savefig(svg_path, dpi=self.dpi, format='svg', bbox_inches='tight')
                print(f"Successfully exported SVG to: {svg_path}")
                print("Note: Excel export requires manual insertion of SVG files")
            else:  # Default to PDF
                plt.savefig(file_path, dpi=self.dpi, format='pdf', bbox_inches='tight')
            print(f"Successfully exported chart to: {file_path}")
        except Exception as e:
            print(f"Failed to export chart to {file_path}: {e}")
        plt.close(self.fig)
    
    def plot_state_and_modality(self, results):
        plt.step(results['time'], results['state'], where='post', linewidth=2, label='Five Hearts State')
        plt.plot(results['time'], results['input_norm'], 'r--', linewidth=1.5, label='Input Intensity')
        plt.title('Five Hearts State and Multimodal Input Intensity')
        plt.ylabel('State/Intensity')
        plt.xlabel('Time Steps')
        plt.yticks([0, 1, 2, 3, 4, 5], ['Idle', 'Initial Response', 'Seeking', 'Deciding', 'Purification', 'Continuous Flow'])
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Mark input unit changes
        prev_unit = ""
        for i, (time_val, unit) in enumerate(zip(results['time'], results['current_unit'])):
            if unit != prev_unit and unit != "":
                plt.axvline(x=time_val, color='g', linestyle=':', alpha=0.5)
                plt.text(time_val, plt.ylim()[1]*0.9, unit, ha='center', fontsize=8, rotation=90)
            prev_unit = unit
    
    def plot_gauge_field_dynamics(self, yang_mills_field):
        if len(yang_mills_field.history['A_mu']) < 2:
            plt.text(0.5, 0.5, 'Insufficient gauge field data', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            return
            
        A_traces = []
        for A in yang_mills_field.history['A_mu']:
            trace = np.trace(A)
            A_traces.append(np.linalg.norm(trace))
            
        plt.plot(range(len(A_traces)), A_traces, 'b-', linewidth=2)
        plt.title('Gauge Potential A_μ Evolution')
        plt.ylabel('Gauge Potential Intensity')
        plt.xlabel('Time Steps')
        plt.grid(True, linestyle='--', alpha=0.7)
    
    def plot_mental_factors_activity(self, mental_factors):
        if not mental_factors.particular_history:
            plt.text(0.5, 0.5, 'Insufficient mental factors data', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            return
            
        particular_data = np.array(mental_factors.particular_history)
        factor_names = ['Desire', 'Understanding', 'Remembrance', 'Concentration', 'Wisdom']
        
        plt.imshow(particular_data.T, cmap='hot', aspect='auto', origin='lower')
        plt.title('Five Specific Mental Factors Activity Heatmap')
        plt.ylabel('Mental Factor Type')
        plt.xlabel('Time Steps')
        plt.yticks(range(len(factor_names)), factor_names)
        plt.colorbar(label='Activity Intensity')
        
        min_val = np.min(particular_data)
        max_val = np.max(particular_data)
        plt.text(0.02, 0.98, f'Intensity Range: [{min_val:.2f}, {max_val:.2f}]', 
                 transform=plt.gca().transAxes, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def plot_seed_evolution(self, results, ax):
        s_data = results['s']
        if len(s_data) == 0:
            ax.text(0.5, 0.5, 0.5, 'Insufficient seed data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            return
            
        x, y, z = s_data[:, 0], s_data[:, 1], s_data[:, 2]
        
        colors = range(len(x))
        scatter = ax.scatter(x, y, z, c=colors, cmap=self.color_map, s=30, alpha=0.7)
        
        ax.plot(x, y, z, 'gray', alpha=0.3)
        
        ax.set_title('Alayavijnana Seed Evolution Trajectory')
        ax.set_xlabel('Seed Dimension 1 (Attention)')
        ax.set_ylabel('Seed Dimension 2 (Concept Activation)')
        ax.set_zlabel('Seed Dimension 3 (Memory Retrieval)')
        
        plt.colorbar(scatter, ax=ax, label='Time Steps')
    
    def plot_manas_kleshas(self, mental_factors):
        if not mental_factors.klesha_history:
            plt.text(0.5, 0.5, 'Insufficient kleshas data', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            return
            
        klesha_data = np.array(mental_factors.klesha_history)
        klesha_names = ['Greed', 'Hatred', 'Ignorance', 'Pride', 'Doubt', 'Wrong Views']
        
        for i, name in enumerate(klesha_names):
            plt.plot(range(len(klesha_data)), klesha_data[:, i], label=name, linewidth=1.5)
            
        plt.title('Manas Six Root Kleshas Intensity')
        plt.ylabel('Intensity')
        plt.xlabel('Time Steps')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        min_val = np.min(klesha_data)
        max_val = np.max(klesha_data)
        plt.text(0.02, 0.98, f'Intensity Range: [{min_val:.2f}, {max_val:.2f}]', 
                 transform=plt.gca().transAxes, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def plot_consciousness_phase_portrait(self, results):
        if len(results['consciousness']) < 2:
            plt.text(0.5, 0.5, 'Insufficient consciousness data', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            return
            
        consciousness = results['consciousness']
        integration = results['integration_strength']
        
        plt.scatter(consciousness, integration, c=range(len(consciousness)), 
                   cmap=self.color_map, alpha=0.6, s=20)
        plt.plot(consciousness, integration, 'k-', alpha=0.3, linewidth=1)
        
        plt.title('Consciousness-Integration Phase Portrait')
        plt.xlabel('Consciousness Intensity')
        plt.ylabel('Conceptual Integration Strength')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.text(0.02, 0.98, 
                 f'Integration Strength Range: [{min(integration):.3f}, {max(integration):.3f}]', 
                 transform=plt.gca().transAxes, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def plot_state_parallel_coordinates(self, results, show=True):
        # plt.figure(figsize=(12, 8)) already called in create_separate_charts
        s_data = results['s']
        if len(s_data) == 0:
            plt.text(0.5, 0.5, 'Insufficient state data', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            if show:
                plt.show()
            return
            
        for i in range(params.STATE_DIM):
            plt.plot(results['time'], s_data[:, i], label=f'Dimension {i+1}', linewidth=1.5, alpha=0.7)
            
        plt.title('Alayavijnana Seed Five-Dimensional State Evolution')
        plt.xlabel('Time')
        plt.ylabel('State Value')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        dim_descriptions = [
            "1: Attention",
            "2: Concept Activation",
            "3: Memory Retrieval",
            "4: Semantic Association 1",
            "5: Semantic Association 2"
        ]
        plt.figtext(0.15, 0.01, "\n".join(dim_descriptions), fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        if show:
            plt.show()
    
    def plot_memory_and_integration(self, results, show=True):
        # plt.figure(figsize=(12, 6)) already called in create_separate_charts
        
        plt.plot(results['time'], results['memory_strength'], 'b-', label='Memory Strength', linewidth=2)
        plt.plot(results['time'], results['integration_strength'], 'r--', label='Conceptual Integration Strength', linewidth=2)
        
        plt.title('Memory Strength vs Conceptual Integration Strength')
        plt.xlabel('Time Steps')
        plt.ylabel('Intensity Value')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        mem_min, mem_max = np.min(results['memory_strength']), np.max(results['memory_strength'])
        int_min, int_max = np.min(results['integration_strength']), np.max(results['integration_strength'])
        
        stats_text = (f"Memory Strength Range: [{mem_min:.3f}, {mem_max:.3f}]\n"
                     f"Integration Strength Range: [{int_min:.3f}, {int_max:.3f}]")
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        if show:
            plt.show()
        
    def plot_detailed_state_transitions(self, results, show=True):
        """Plot detailed state transitions analysis"""
        # plt.figure(figsize=(15, 10)) already called in create_separate_charts
        
        # 1. State duration distribution
        plt.subplot(2, 2, 1)
        state_durations = []
        current_state = results['state'][0]
        duration = 1
        state_names = ['Idle', 'Initial Response', 'Seeking', 'Deciding', 'Purification', 'Continuous Flow']
        
        for i in range(1, len(results['state'])):
            if results['state'][i] == current_state:
                duration += 1
            else:
                state_durations.append((current_state, duration))
                current_state = results['state'][i]
                duration = 1
        state_durations.append((current_state, duration))
        
        state_counts = {state: 0 for state in range(6)}
        for state, duration in state_durations:
            state_counts[state] += duration
            
        states = list(state_counts.keys())
        durations = [state_counts[state] for state in states]
        state_labels = [state_names[state] for state in states]
        
        plt.bar(state_labels, durations, color=plt.cm.Set3(np.linspace(0, 1, 6)))
        plt.title('State Duration Distribution')
        plt.ylabel('Time Steps')
        plt.xticks(rotation=45)
        
        # 2. State transition frequency
        plt.subplot(2, 2, 2)
        transition_matrix = np.zeros((6, 6))
        for i in range(len(results['state'])-1):
            from_state = int(results['state'][i])
            to_state = int(results['state'][i+1])
            if from_state != to_state:
                transition_matrix[from_state, to_state] += 1
        
        plt.imshow(transition_matrix, cmap='YlOrRd', aspect='auto')
        plt.title('State Transition Frequency Heatmap')
        plt.xlabel('Target State')
        plt.ylabel('Source State')
        plt.xticks(range(6), state_labels, rotation=45)
        plt.yticks(range(6), state_labels)
        plt.colorbar(label='Transition Count')
        
        # 3. Input intensity vs state relationship
        plt.subplot(2, 2, 3)
        for state in range(6):
            state_mask = results['state'] == state
            if np.any(state_mask):
                state_inputs = results['input_norm'][state_mask]
                plt.hist(state_inputs, alpha=0.6, label=state_names[state], bins=20)
        plt.title('Input Intensity Distribution by State')
        plt.xlabel('Input Intensity')
        plt.ylabel('Frequency')
        plt.legend()
        
        # 4. Integration strength vs state relationship
        plt.subplot(2, 2, 4)
        state_integration_means = []
        for state in range(6):
            state_mask = results['state'] == state
            if np.any(state_mask):
                mean_integration = np.mean(results['integration_strength'][state_mask])
                state_integration_means.append(mean_integration)
            else:
                state_integration_means.append(0)
                
        plt.bar(state_names, state_integration_means, color='lightblue')
        plt.title('Average Conceptual Integration Strength by State')
        plt.ylabel('Average Integration Strength')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        if show:
            plt.show()

# 九、改进的仿真函数
def run_advanced_simulation(multi_modal_input, yang_mills_field, mental_factors, manas_mapping, total_time=140):
    """运行高级仿真"""
    time_points = np.linspace(0, total_time, total_time)
    s = np.array([0.2, 0.2, 0.3, 0.2, 0.2])
    
    memory = RobustShortTermMemory(capacity=5)
    # 修复：使用正确的类名实例化
    state_machine = ImprovedFiveHeartsStateMachine()
    
    results = {
        'time': time_points,
        'state': np.zeros(total_time),
        's': np.zeros((total_time, params.STATE_DIM)),
        'consciousness': np.zeros(total_time),
        'feeling': np.zeros(total_time),
        'input_norm': np.zeros(total_time),
        'memory_strength': np.zeros(total_time),
        'integration_strength': np.zeros(total_time),
        'current_unit': [''] * total_time,
        'memory_capacity': np.zeros(total_time),
        'learning_rate': np.zeros(total_time),
        'modalities_weights': np.zeros((total_time, len(multi_modal_input.modalities)))
    }
    
    modality_list = list(multi_modal_input.modalities.keys())
    prev_auditory_item = None
    prev_integration = 0  # 新增：用于平滑滤波的前一个整合强度
    
    for i, t in enumerate(time_points):
        # 获取当前输入
        current_input = multi_modal_input.get_input(t)
        
        # 检测输入变化
        current_auditory_item = multi_modal_input.get_current_item(ModalityType.AUDITORY)
        input_changed = (current_auditory_item != prev_auditory_item and prev_auditory_item is not None)
        prev_auditory_item = current_auditory_item
        
        # 应用末那识执取
        s_transformed = manas_mapping.apply_manas_effect(s, 'volition', 0.5)
        
        # 检索记忆
        memory_content = memory.retrieve_memory(t)
        
        # 动态概念整合
        # 修复：使用正确的函数名并传递必要的参数
        integrated_concept, integration_strength = dynamic_conceptual_integration(
            current_input, memory_content, t, prev_integration)
        prev_integration = integration_strength  # 更新前一个整合强度
        # 在计算心所函数前添加规范场变换
        s = yang_mills_field.apply_to_state(s)
    
    # 然后继续现有计算...
        universal = mental_factors.universal_factors(s, current_input, memory, t)
        # 计算心所函数
        def touch_function(s, input):
            return np.dot(s, input) * 0.5
            
        def attention_function(s, touch):
            return np.linalg.norm(s) * (0.3 + touch)
            
        def feeling_function(s, touch, attention):
            return np.tanh(touch * attention)
            
        def perception_function(touch, attention):
            return np.array([touch * 0.5, attention * 0.5, touch * attention])
            
        def volition_function(s, perception, feeling):
            return np.tanh(s * (0.3 + feeling))
        
        T = touch_function(s, current_input)
        A = attention_function(s, T)
        V = feeling_function(s, T, A)
        P = perception_function(T, A)
        C = volition_function(s, P, V)
        
        # 添加记忆
        memory.add_memory(integrated_concept, t, importance=0.1 + integration_strength)
        
        # 记录记忆强度
        results['memory_strength'][i] = memory.get_memory_strength()
        results['integration_strength'][i] = integration_strength
        
        # 计算心所
        universal = mental_factors.universal_factors(s, current_input, memory, t)
        contact, attention, feeling, perception, volition = universal
        
        particular = mental_factors.particular_factors(s, perception, memory, t)
        desire, resolution, mindfulness, concentration, wisdom = particular
        
        virtuous = mental_factors.virtuous_factors(wisdom, perception)
        kleshas = mental_factors.klesha_factors(s, perception)
        mental_factors.update_secondary_factors(kleshas)
        
        yang_mills_field.update_current(s, mental_factors)
        yang_mills_field.yang_mills_equation()
        
        heart_functions = (contact, attention, feeling, perception, volition, mindfulness, wisdom)
        
        # 更新状态机，传入输入变化信息
        state, feeling_value = state_machine.update(
            s, current_input, heart_functions, t, integration_strength, input_changed)
        
        learning_rate = feedback_mechanism(s, state, integration_strength, memory)
        results['learning_rate'][i] = learning_rate
        results['memory_capacity'][i] = memory.capacity
        
        # 印记效应
        if state in [3, 4] and integration_strength > 0.2 and mindfulness > 0.1:
            s = imprint_effect(s, feeling_value, learning_rate)
            memory.add_memory(integrated_concept, t, importance=mindfulness)
        
        # 记录当前状态
        results['state'][i] = state
        results['s'][i] = s
        results['consciousness'][i] = np.linalg.norm(perception)
        results['feeling'][i] = feeling_value
        results['memory_strength'][i] = memory.get_memory_strength()
        results['input_norm'][i] = np.linalg.norm(current_input)
        
        # 获取当前输入单元
        if ModalityType.AUDITORY in multi_modal_input.modalities:
            current_unit = multi_modal_input.get_current_item(ModalityType.AUDITORY)
            results['current_unit'][i] = current_unit if current_unit is not None else ""
        
        if i < total_time - 1:
            dsdt = alaya_evolution(s, t, current_input, heart_functions, learning_rate)
            s = s + dsdt * (time_points[1] - time_points[0])
            s = np.clip(s, -1, 1)
    
    return results, state_machine

# 十、改进的分析函数
def analyze_state_transitions(results, state_machine, multi_modal_input):
    """Detailed state transitions analysis"""
    print(f"DEBUG: multi_modal_input type: {type(multi_modal_input)}")
    print(f"DEBUG: multi_modal_input content: {multi_modal_input}")
    print("\nGuanju Cognitive Process Detailed Analysis:")
    print(f"Input sequence length: {multi_modal_input.sequence_length} sentences")
    print(f"Simulation total time steps: {len(results['time'])}")
    
    # Analyze state transitions
    state_names = ['Idle', 'Initial Response', 'Seeking', 'Deciding', 'Purification', 'Continuous Flow']
    
    print(f"\nState machine transition history ({len(state_machine.transition_history)} transitions):")
    for t, from_state, to_state in state_machine.transition_history:
        from_name = state_names[int(from_state)] if 0 <= from_state < len(state_names) else "Unknown"
        to_name = state_names[int(to_state)] if 0 <= to_state < len(state_names) else "Unknown"
        
        # Fix index out of bounds issue
        time_idx = min(int(round(t)), len(results['current_unit'])-1)
        current_unit = results['current_unit'][time_idx]
        
        print(f"Time {t:.1f}: {from_name} -> {to_name} (Input: '{current_unit}')")
    
    # Calculate state durations
    state_durations = {state: 0 for state in range(6)}
    current_state = results['state'][0]
    duration = 1
    
    for i in range(1, len(results['state'])):
        if results['state'][i] == current_state:
            duration += 1
        else:
            state_durations[current_state] += duration
            current_state = results['state'][i]
            duration = 1
    state_durations[current_state] += duration
    
    print(f"\nState durations:")
    for state in range(6):
        print(f"{state_names[state]}: {state_durations[state]} time steps")
    
    # Find state changes for each input unit
    print(f"\nCognitive states for each sentence:")
    unique_units = []
    for unit in results['current_unit']:
        if unit and unit not in unique_units:
            unique_units.append(unit)
    
    for unit in unique_units:
        unit_indices = [i for i, u in enumerate(results['current_unit']) if u == unit]
        if unit_indices:
            states_in_unit = [results['state'][i] for i in unit_indices]
            unique_states = set(states_in_unit)
            state_names_in_unit = [state_names[int(s)] for s in unique_states]
            print(f"'{unit}': {', '.join(state_names_in_unit)}")
    
    # Find integration strength peak
    max_integration = np.max(results['integration_strength'])
    max_integration_time = np.argmax(results['integration_strength'])
    max_integration_unit = results['current_unit'][max_integration_time]
    
    print(f"\nCognitive integration analysis:")
    print(f"Maximum conceptual integration strength: {max_integration:.3f} (Time: {max_integration_time})")
    print(f"Corresponding sentence: '{max_integration_unit}'")
    print(f"Corresponding state: {state_names[int(results['state'][max_integration_time])]}")
    
    return state_durations

# 11. Run comprehensive test
def run_comprehensive_simulation(export_format='pdf'):
    """Run comprehensive simulation test"""
    
    # 1. Initialize Guanjue multi-modal input
    processor = SimpleClassicsProcessor()
    guanju_text = processor.create_guanju_input()
    
    # Create richer visual and conceptual features
    visual_features = []
    conceptual_features = []
    
    # Create features for each sentence
    for i, line in enumerate(guanju_text):
        # Visual features: based on emotional color of the sentence
        if i in [0, 1]:  # Opening scene
            visual_features.append(np.array([0.8, 0.7, 0.9, 0.6, 0.8]))
        elif i in [2, 3, 4]:  # Pursuit process
            visual_features.append(np.array([0.6, 0.8, 0.7, 0.5, 0.9]))
        elif i in [5]:  # Longing and anxiety
            visual_features.append(np.array([0.4, 0.6, 0.5, 0.7, 0.6]))
        else:  # Harmonious ending
            visual_features.append(np.array([0.9, 0.8, 0.8, 0.7, 0.7]))
        
        # Conceptual features: moral connotation
        if i in [0, 1]:  # Natural harmony
            conceptual_features.append(np.array([0.9, 0.8, 0.7, 0.6, 0.8]))
        elif i in [2, 3, 4]:  # Moderate pursuit
            conceptual_features.append(np.array([0.7, 0.8, 0.9, 0.8, 0.7]))
        elif i in [5]:  # Emotional restraint
            conceptual_features.append(np.array([0.6, 0.7, 0.8, 0.9, 0.6]))
        else:  # Ritual norms
            conceptual_features.append(np.array([0.8, 0.7, 0.8, 0.9, 0.8]))
    
    modalities = {
        ModalityType.AUDITORY: guanju_text,
        ModalityType.VISUAL: visual_features,
        ModalityType.CONCEPTUAL: conceptual_features
    }
    multi_modal_input = MultiModalInput(modalities, sequence_duration=3)
    multi_modal_input.optimize_modality_weights()
    
    # 2. Initialize system components
    yang_mills_field = YangMillsField()
    mental_factors = CompleteMentalFactors()
    manas_mapping = ManasGaugeMapping()
    viz = AdvancedVisualization(export_format=export_format)
    
    # 3. Run simulation
    print("Starting Buddhist cognition field model simulation...")
    results, state_machine = run_advanced_simulation(
        multi_modal_input, 
        yang_mills_field, 
        mental_factors, 
        manas_mapping,
        total_time=140
    )
    print("Simulation completed, generating visualization results...")
    
    # 4. Detailed state transitions analysis
    state_durations = analyze_state_transitions(results, state_machine, multi_modal_input)
    
    # 5. Generate Guanjue analysis visualization
    viz.create_comprehensive_dashboard(results, yang_mills_field, mental_factors)
    
    # 6. Export individual charts
    print("=== Starting individual charts export ===")
    print(f"Current export format: {viz.export_format}")
    print(f"Current export DPI: {viz.dpi}")
    try:
        viz.create_separate_charts(results, yang_mills_field, mental_factors)
        print("=== Individual charts export completed ===")
    except Exception as e:
        print(f"=== Individual charts export failed: {e} ===")
    
    return results, state_machine, multi_modal_input

# Run comprehensive test
if __name__ == "__main__":
    np.random.seed(42)
    
    print("=== Buddhist Cognition Model - Guanjue Analysis ===")
    # You can change the export format here: 'pdf', 'tiff', or 'excel'
    export_format = 'pdf'  # Change to 'tiff' or 'excel' as needed
    results, state_machine, multi_modal_input = run_comprehensive_simulation(export_format=export_format)
