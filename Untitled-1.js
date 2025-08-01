/* ===== JavaScript文件 - Multi-Agent问答平台 ===== */
/* 这个文件包含了问答平台的所有交互逻辑和功能实现 */
/* 包括消息发送、历史记录管理、UI更新、文件上传等功能 */

/* ===== 全局变量定义 ===== */
/* 用于存储应用程序的全局状态和配置 */
let sessionId = null;                    // 当前会话ID，用于标识用户会话
const API_BASE_URL = 'http://localhost:8000';  // 后端API的基础URL地址

/**
 * 历史对话记录管理类
 * 负责管理用户的历史对话记录，包括保存、获取、删除等功能
 * 使用localStorage进行本地存储，最多保留4次对话记录
 */
class ConversationHistory {
    /**
     * 构造函数 - 初始化历史记录管理器
     * 设置最大历史记录数量、存储键名和当前对话数组
     */
    constructor() {
        this.maxHistory = 4;                 // 最多保留4次对话记录，避免存储过多数据
        this.storageKey = 'chat_history';    // localStorage存储键名
        this.currentConversation = [];       // 当前对话的消息数组
    }
    
    /**
     * 保存当前对话到历史记录
     * 将当前进行中的对话保存到localStorage中
     * 自动生成对话标题和时间戳，并维护最大记录数量限制
     */
    saveCurrentConversation() {
        // 如果当前对话为空，则不保存
        if (this.currentConversation.length === 0) return;
        
        // 创建对话记录对象
        const conversation = {
            id: Date.now(),                      // 使用时间戳作为唯一ID
            timestamp: new Date().toISOString(), // ISO格式的时间戳
            messages: [...this.currentConversation], // 复制当前对话消息数组
            title: this.generateConversationTitle()  // 自动生成对话标题
        };
        
        // 获取现有历史记录
        let history = this.getHistory();
        // 将新对话添加到数组开头（最新的在前面）
        history.unshift(conversation);
        
        // 保持最多4次记录，删除超出的旧记录
        if (history.length > this.maxHistory) {
            history = history.slice(0, this.maxHistory);
        }
        
        // 保存到localStorage
        localStorage.setItem(this.storageKey, JSON.stringify(history));
        console.log('对话已保存到历史记录');  // 调试信息
    }
    
    /**
     * 获取历史记录
     * 从localStorage中读取并解析历史对话记录
     * @returns {Array} 历史对话记录数组，如果没有记录则返回空数组
     */
    getHistory() {
        const stored = localStorage.getItem(this.storageKey);
        return stored ? JSON.parse(stored) : [];  // 解析JSON或返回空数组
    }
    
    /**
     * 添加消息到当前对话
     * 将用户消息或AI回复添加到当前对话记录中
     * @param {string} type - 消息类型（'user' 或 'ai'）
     * @param {string} content - 消息内容
     */
    addMessage(type, content) {
        // 创建消息对象并添加到当前对话数组
        this.currentConversation.push({
            type: type,                          // 消息类型：'user'（用户）或'ai'（AI回复）
            content: content,                    // 消息内容文本
            timestamp: new Date().toISOString()  // 消息时间戳，ISO格式
        });
    }
    
    /**
     * 清空当前对话
     * 重置当前对话数组，用于开始新的对话
     */
    clearCurrentConversation() {
        this.currentConversation = [];  // 清空当前对话消息数组
    }
    
    /**
     * 生成对话标题
     * 自动生成对话标题，取第一条用户消息的前20个字符作为标题
     * 如果超过20个字符则添加省略号
     * @returns {string} 生成的对话标题
     */
    generateConversationTitle() {
        // 查找第一条用户消息
        const firstUserMessage = this.currentConversation.find(msg => msg.type === 'user');
        if (firstUserMessage) {
            // 截取前20个字符作为标题
            const title = firstUserMessage.content.substring(0, 20);
            // 如果原文本更长，则添加省略号
            return title.length < firstUserMessage.content.length ? title + '...' : title;
        }
        return '新对话';  // 如果没有用户消息，返回默认标题
    }
    
    /**
     * 加载指定的历史对话
     * 根据对话ID从历史记录中加载指定对话，并在界面上重新显示
     * @param {number} conversationId - 要加载的对话ID
     */
    loadConversation(conversationId) {
        const history = this.getHistory();  // 获取所有历史记录
        // 根据ID查找指定对话
        const conversation = history.find(conv => conv.id === conversationId);
        if (conversation) {
            // 清空当前界面的对话容器
            const container = document.getElementById('qaContainer');
            container.innerHTML = '';
            
            // 重新显示历史对话中的所有消息
            conversation.messages.forEach(msg => {
                // false参数表示不将这些消息添加到当前对话记录中（避免重复）
                addMessageToUI(msg.type, msg.content, false);
            });
            
            // 将加载的对话设置为当前对话（复制数组避免引用问题）
            this.currentConversation = [...conversation.messages];
        }
    }
    
    /**
     * 删除指定的历史对话
     * 从localStorage中删除指定ID的历史对话记录
     * @param {number} conversationId - 要删除的对话ID
     */
    deleteConversation(conversationId) {
        let history = this.getHistory();  // 获取当前历史记录
        // 过滤掉指定ID的对话，保留其他对话
        history = history.filter(conv => conv.id !== conversationId);
        // 更新localStorage中的历史记录
        localStorage.setItem(this.storageKey, JSON.stringify(history));
    }
}

/* ===== 全局实例创建 ===== */
// 创建历史记录管理实例，用于整个应用程序的对话历史管理
const conversationHistory = new ConversationHistory();

/* ===== 消息发送和处理函数 ===== */

/**
 * 发送消息到后端API
 * 主要的消息发送函数，处理用户输入并调用后端API
 * 包括输入验证、UI更新、错误处理等功能
 */
async function sendMessage() {
    // 获取输入框元素和用户输入的消息
    const input = document.getElementById('messageInput');
    const message = input.value.trim();  // 去除首尾空格
    
    // 如果消息为空，则不发送
    if (!message) return;
    
    // 清空输入框，为下次输入做准备
    input.value = '';
    
    // 立即在界面上显示用户消息，提供即时反馈
    addMessageToUI('user', message);
    
    try {
        // 调用RAG流式API获取AI响应
        await streamRAGResponse(message);
    } catch (error) {
        // 错误处理：记录错误并向用户显示友好的错误信息
        console.error('发送消息失败:', error);
        addMessageToUI('error', '抱歉，发送消息时出现错误，请稍后重试。');
    }
}

/**
 * 使用SSE流式接收RAG响应
 * 通过Server-Sent Events (SSE) 技术实现流式响应
 * 调用FastAPI的/query接口获取RAG回答
 * @param {string} message - 用户发送的消息内容
 */
async function streamRAGResponse(message) {
    // 使用FastAPI的流式查询接口
    const endpoint = '/query';
    
    // 构建请求体，包含用户消息
    const requestBody = {
        query: message  // 用户输入的消息内容
    };
    
    // 发送POST请求到流式查询API端点
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',                          // 使用POST方法
        headers: {
            'Content-Type': 'application/json'   // 设置请求内容类型为JSON
        },
        body: JSON.stringify(requestBody)        // 将请求体转换为JSON字符串
    });
    
    // 检查HTTP响应状态，如果不成功则抛出错误
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    // 获取响应流的读取器和文本解码器
    const reader = response.body.getReader();  // 用于读取流式数据
    const decoder = new TextDecoder();         // 用于将字节流解码为文本
    let aiMessageElement = null;  // 用于存储AI消息的DOM元素引用
    let aiCompleteMessage = '';   // 用于累积完整的AI回答内容
    
    try {
        // 持续读取流式数据直到完成
        while (true) {
            // 从流中读取数据块
            const { done, value } = await reader.read();
            if (done) {
                // 流式传输完成，将完整的AI回答保存到历史记录
                if (aiCompleteMessage.trim()) {
                    conversationHistory.addMessage('ai', aiCompleteMessage.trim());
                    console.log('AI完整回答已保存到历史记录');
                }
                break;  // 退出循环
            }
            
            // 将字节数据解码为文本
            const chunk = decoder.decode(value);
            // 按行分割数据（SSE格式是按行传输的）
            const lines = chunk.split('\n');
            
            // 处理每一行数据
            for (const line of lines) {
                // 检查是否是SSE数据行（以'data: '开头）
                if (line.startsWith('data: ')) {
                    // 提取实际的数据内容（去掉'data: '前缀）
                    const data = line.slice(6);
                    
                    // 检查是否是结束标志
                    if (data === '[DONE]') {
                        // 流式传输完成，将完整的AI回答保存到历史记录
                        if (aiCompleteMessage.trim()) {
                            conversationHistory.addMessage('ai', aiCompleteMessage.trim());
                            console.log('AI完整回答已保存到历史记录');
                        }
                        return;  // 流式传输完成，退出函数
                    }
                    
                    try {
                        // 解析JSON数据
                        const parsed = JSON.parse(data);
                        
                        // 处理FastAPI流式响应格式
                        if (parsed.delta !== undefined) {
                            // 如果是第一个数据块，创建AI消息元素
                            if (!aiMessageElement) {
                                aiMessageElement = addMessageToUI('ai', '', false);
                                aiCompleteMessage = '';  // 重置完整消息累积器
                                console.log('AI开始回复 (RAG模式)');
                            }
                            
                            // 接收AI回复的文本片段，逐步追加到消息元素和完整消息
                            if (parsed.delta && !parsed.finished) {
                                const chunkContent = parsed.delta;
                                aiMessageElement.textContent += chunkContent;
                                aiCompleteMessage += chunkContent;  // 累积完整消息
                            }
                            
                            // 检查是否完成
                            if (parsed.finished) {
                                // AI回复结束，完成消息处理并保存到历史记录
                                if (aiCompleteMessage.trim()) {
                                    conversationHistory.addMessage('ai', aiCompleteMessage.trim());
                                    console.log('AI回复结束，完整回答已保存到历史记录 (RAG模式)');
                                }
                                return;
                            }
                        } else if (parsed.error) {
                            // 接收错误消息并显示给用户
                            const errorContent = parsed.error || '发生未知错误';
                            addMessageToUI('error', errorContent);
                            console.error('接收到错误消息:', errorContent);
                            return;
                        }
                    } catch (parseError) {
                        // JSON解析错误处理
                        console.error('解析SSE数据失败:', parseError);
                    }
                }
            }
        }
    } finally {
        // 确保释放流读取器的锁定，避免资源泄漏
        reader.releaseLock();
    }
}

/* ===== UI界面更新函数 ===== */

/**
 * 添加消息到UI界面（聊天气泡样式）
 * 这是核心的UI更新函数，负责在界面上显示用户消息、AI回复和错误信息
 * 支持不同类型的消息样式和自动滚动到最新消息
 * @param {string} type - 消息类型 ('user', 'ai', 'error')
 * @param {string} content - 消息内容
 * @param {boolean} addToHistory - 是否添加到历史记录，默认为true
 * @returns {HTMLElement} 返回创建的消息元素（AI消息返回内容元素，其他返回整个消息容器）
 */
function addMessageToUI(type, content, addToHistory = true) {
    // 获取消息容器元素
    const container = document.getElementById('qaContainer');
    // 创建消息容器div
    const messageDiv = document.createElement('div');
    // 设置CSS类名，包含通用类和类型特定类
    messageDiv.className = `chat-message ${type}`;
    
    // 添加到历史记录（如果需要且是有效的消息类型）
    if (addToHistory && (type === 'user' || type === 'ai')) {
        conversationHistory.addMessage(type, content);
    }
    
    // 根据消息类型创建不同的UI结构
    if (type === 'user') {
        // 用户消息：右侧对齐，显示用户名和头像
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="message-info">chunkit</div>
                <div class="message-bubble">${formatMessageContent(content)}</div>
            </div>
            <div class="message-avatar user"></div>
        `;
    } else if (type === 'ai') {
        // AI消息：左侧对齐，显示AI标识和助手名称
        // 创建独立的答案div，用于流式更新内容
        const answerDiv = document.createElement('div');
        answerDiv.className = 'message-bubble';
        answerDiv.innerHTML = formatMessageContent(content);
        
        // 构建AI消息的基本结构
        messageDiv.innerHTML = `
            <div class="message-avatar ai">AI</div>
            <div class="message-content">
                <div class="message-info">RAG助手</div>
            </div>
        `;
        
        // 将答案div添加到消息内容中
        messageDiv.querySelector('.message-content').appendChild(answerDiv);
        // 立即添加到容器并滚动到视图（用于流式显示）
        container.appendChild(messageDiv);
        messageDiv.scrollIntoView({ behavior: 'smooth' });
        
        // 返回答案div，用于后续的流式内容更新
        return answerDiv;
    } else if (type === 'error') {
        // 错误消息：特殊样式，红色主题
        messageDiv.innerHTML = `
            <div class="message-avatar ai">⚠️</div>
            <div class="message-content">
                <div class="message-info">系统提示</div>
                <div class="message-bubble" style="background: #ffebee; color: #c62828; border-color: #ef5350;">${content}</div>
            </div>
        `;
    }
    
    // 将消息添加到容器中
    container.appendChild(messageDiv);
    // 平滑滚动到最新消息，确保用户能看到最新内容
    messageDiv.scrollIntoView({ behavior: 'smooth' });
    
    // 返回整个消息div元素
    return messageDiv;
}

/* ===== 消息内容格式化函数 ===== */

/**
 * 初始化markdown渲染器配置
 * 配置marked.js和highlight.js的选项
 */
function initializeMarkdownRenderer() {
    // 配置marked.js选项
    marked.setOptions({
        highlight: function(code, lang) {
            // 如果指定了语言且highlight.js支持，则进行代码高亮
            if (lang && hljs.getLanguage(lang)) {
                try {
                    return hljs.highlight(code, { language: lang }).value;
                } catch (err) {
                    console.warn('代码高亮失败:', err);
                }
            }
            // 否则进行自动检测高亮
            try {
                return hljs.highlightAuto(code).value;
            } catch (err) {
                console.warn('自动代码高亮失败:', err);
                return code;
            }
        },
        breaks: true,        // 支持换行符转换为<br>
        gfm: true,          // 启用GitHub风格的markdown
        tables: true,       // 支持表格
        sanitize: false,    // 不清理HTML（需要显示代码高亮）
        smartypants: true   // 智能标点符号转换
    });
}

/**
 * 格式化消息内容，支持markdown渲染和特殊格式处理
 * 使用正则表达式识别特定模式并添加样式，提升消息的可读性
 * @param {string} content - 原始消息内容
 * @returns {string} 格式化后的HTML内容
 */
function formatMessageContent(content) {
    try {
        // 首先进行markdown渲染
        let htmlContent = marked.parse(content);
        
        // 然后处理自定义的文件引用格式 (例如: styles.css 105-120)
        // 匹配"文件名.扩展名 数字-数字"的模式，添加文件图标和特殊样式
        htmlContent = htmlContent.replace(
            /(\w+\.\w+)\s+(\d+-\d+)/g,
            '<span class="file-reference"><span class="file-icon">📄</span>$1 $2</span>'
        );
        
        // 处理关键词高亮（可选功能）
        // 匹配一些常见的技术关键词并添加高亮样式
        const keywords = ['API', 'HTTP', 'JSON', 'CSS', 'HTML', 'JavaScript', 'Python', 'FastAPI', 'RAG'];
        keywords.forEach(keyword => {
            const regex = new RegExp(`\\b${keyword}\\b`, 'gi');
            htmlContent = htmlContent.replace(regex, `<span class="keyword-highlight">${keyword}</span>`);
        });
        
        return htmlContent;
    } catch (error) {
        console.error('格式化消息内容失败:', error);
        // 如果格式化失败，返回原始内容
        return content;
    }
}

/* ===== 页面初始化和事件监听 ===== */

/**
 * 页面加载完成后的初始化函数
 * 设置事件监听器和初始化配置
 */
document.addEventListener('DOMContentLoaded', function() {
    // 初始化markdown渲染器
    initializeMarkdownRenderer();
    
    // 为输入框添加回车键发送消息的功能
    const messageInput = document.getElementById('messageInput');
    if (messageInput) {
        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();  // 阻止默认的换行行为
                sendMessage();       // 发送消息
            }
        });
    }
    
    console.log('RAG问答平台初始化完成');
});

/* ===== 其他功能函数 ===== */

/**
 * 开始新对话
 * 清空当前对话界面和历史记录
 */
function startNewConversation() {
    // 保存当前对话到历史记录
    conversationHistory.saveCurrentConversation();
    
    // 清空界面
    const container = document.getElementById('qaContainer');
    container.innerHTML = '';
    
    // 清空当前对话记录
    conversationHistory.clearCurrentConversation();
    
    console.log('开始新对话');
}

/**
 * 显示历史记录弹窗
 * 创建并显示历史对话记录的模态窗口
 */
function showHistory() {
    const history = conversationHistory.getHistory();
    
    // 创建模态窗口HTML
    const modalHTML = `
        <div class="history-modal" id="historyModal">
            <div class="history-modal-overlay" onclick="closeHistory()"></div>
            <div class="history-modal-content">
                <div class="history-header">
                    <h3>历史对话记录</h3>
                    <button class="close-btn" onclick="closeHistory()">×</button>
                </div>
                <div class="history-list">
                    ${history.length === 0 ? 
                        '<p style="text-align: center; color: #666; padding: 20px;">暂无历史记录</p>' :
                        history.map(conv => `
                            <div class="history-item">
                                <div class="history-item-content">
                                    <div class="history-title">${conv.title}</div>
                                    <div class="history-time">${new Date(conv.timestamp).toLocaleString()}</div>
                                    <div class="history-preview">${conv.messages.length} 条消息</div>
                                </div>
                                <div class="history-actions">
                                    <button class="load-btn" onclick="loadHistoryConversation(${conv.id})">加载</button>
                                    <button class="delete-btn" onclick="deleteHistoryConversation(${conv.id})">删除</button>
                                </div>
                            </div>
                        `).join('')
                    }
                </div>
            </div>
        </div>
    `;
    
    // 添加到页面
    document.body.insertAdjacentHTML('beforeend', modalHTML);
}

/**
 * 关闭历史记录弹窗
 */
function closeHistory() {
    const modal = document.getElementById('historyModal');
    if (modal) {
        modal.remove();
    }
}

/**
 * 加载历史对话
 * @param {number} conversationId - 对话ID
 */
function loadHistoryConversation(conversationId) {
    // 保存当前对话
    conversationHistory.saveCurrentConversation();
    
    // 加载指定对话
    conversationHistory.loadConversation(conversationId);
    
    // 关闭弹窗
    closeHistory();
    
    console.log('已加载历史对话:', conversationId);
}

/**
 * 删除历史对话
 * @param {number} conversationId - 对话ID
 */
function deleteHistoryConversation(conversationId) {
    if (confirm('确定要删除这条历史记录吗？')) {
        conversationHistory.deleteConversation(conversationId);
        // 重新显示历史记录列表
        closeHistory();
        showHistory();
        console.log('已删除历史对话:', conversationId);
    }
}

/**
 * 触发文件上传
 * 目前为占位函数，可以根据需要实现文件上传功能
 */
function triggerFileUpload() {
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.click();
    }
}

/**
 * 退出登录
 * 返回登录页面
 */
function logout() {
    if (confirm('确定要退出登录吗？')) {
        // 保存当前对话
        conversationHistory.saveCurrentConversation();
        
        // 跳转到登录页面
        window.location.href = 'login.html';
    }
}