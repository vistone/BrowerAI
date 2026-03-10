/**
 * ============================================================
 * Google Earth 本地文件系统 API - ES6+ 重构版本
 * ============================================================
 * 
 * 此文件是基于反混淆分析完全重写的版本
 * 使用现代 JavaScript 语法，保持原始功能
 * 
 * 原始文件: google_earth_local_fs.js
 * 混淆工具: Google Closure Compiler (ADVANCED_OPTIMIZATIONS)
 * 
 * @module LocalFileSystem
 * @version 1.0.0
 * @author Google LLC (原始) / BrowerAI (重构)
 */

// ============================================================
// 类型定义 (TypeScript 风格注释)
// ============================================================

/**
 * @typedef {Object} FileMetadata
 * @property {string} name - 文件名
 * @property {number} size - 文件大小 (字节)
 * @property {Date} lastModified - 最后修改时间
 * @property {string} mimeType - MIME 类型
 */

/**
 * @typedef {Object} FileSystemBackend
 * @property {Function} init - 初始化
 * @property {Function} readFile - 读取文件
 * @property {Function} writeFile - 写入文件
 * @property {Function} modifyFile - 修改文件
 * @property {Function} removeFile - 删除文件
 * @property {Function} listFiles - 列出文件
 * @property {Function} getMetadata - 获取元数据
 */

// ============================================================
// 常量
// ============================================================

/** KMZ 文件 MIME 类型 */
const MIME_TYPE_KMZ = "application/vnd.google-earth.kmz";

/** 本地存储键名 */
const STORAGE_KEY_PERMISSION = "fileSystemPermissionGranted";

// ============================================================
// 工具函数
// ============================================================

/**
 * 获取全局对象 (兼容 Node.js, Browser, Worker)
 * @returns {Object} 全局对象
 */
function getGlobalObject() {
    if (typeof globalThis !== 'undefined') return globalThis;
    if (typeof window !== 'undefined') return window;
    if (typeof self !== 'undefined') return self;
    if (typeof global !== 'undefined') return global;
    throw new Error("Cannot find global object");
}

const GLOBAL = getGlobalObject();

// ============================================================
// IndexedDB 后端实现
// ============================================================

/**
 * IndexedDB 文件系统后端
 * 用于不支持 File System Access API 的浏览器
 */
class IndexedDBBackend {
    /**
     * @param {Function} logError - 错误日志函数
     */
    constructor(logError) {
        /** @type {Function} */
        this.logError = logError;
        
        /** @type {IDBDatabase|null} */
        this.database = null;
        
        /** @type {string} */
        this.dbName = 'google-earth-files';
        
        /** @type {string} */
        this.storeName = 'files';
    }
    
    /**
     * 初始化数据库
     * @returns {Promise<void>}
     */
    async initialize() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.dbName, 1);
            
            request.onerror = () => {
                this.logError('IndexedDB open failed');
                reject(request.error);
            };
            
            request.onsuccess = () => {
                this.database = request.result;
                resolve();
            };
            
            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                if (!db.objectStoreNames.contains(this.storeName)) {
                    db.createObjectStore(this.storeName, { keyPath: 'path' });
                }
            };
        });
    }
    
    /**
     * 读取文件
     * @param {string} path - 文件路径
     * @returns {Promise<ArrayBuffer>}
     */
    async readFile(path) {
        return new Promise((resolve, reject) => {
            const transaction = this.database.transaction([this.storeName], 'readonly');
            const store = transaction.objectStore(this.storeName);
            const request = store.get(path);
            
            request.onsuccess = () => {
                if (request.result) {
                    resolve(request.result.content);
                } else {
                    reject(new Error(`File not found: ${path}`));
                }
            };
            
            request.onerror = () => reject(request.error);
        });
    }
    
    /**
     * 写入文件
     * @param {string} path - 文件路径
     * @param {string} mimeType - MIME 类型
     * @param {ArrayBuffer} content - 文件内容
     * @returns {Promise<void>}
     */
    async writeFile(path, mimeType, content) {
        return new Promise((resolve, reject) => {
            const transaction = this.database.transaction([this.storeName], 'readwrite');
            const store = transaction.objectStore(this.storeName);
            
            const fileRecord = {
                path,
                mimeType,
                content,
                lastModified: new Date().toISOString()
            };
            
            const request = store.put(fileRecord);
            
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }
    
    /**
     * 修改文件 (与 writeFile 相同)
     */
    async modifyFile(path, mimeType, content) {
        return this.writeFile(path, mimeType, content);
    }
    
    /**
     * 删除文件
     * @param {string} path - 文件路径
     * @returns {Promise<void>}
     */
    async removeFile(path) {
        return new Promise((resolve, reject) => {
            const transaction = this.database.transaction([this.storeName], 'readwrite');
            const store = transaction.objectStore(this.storeName);
            const request = store.delete(path);
            
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }
    
    /**
     * 列出所有文件
     * @returns {Promise<string[]>}
     */
    async listFiles() {
        return new Promise((resolve, reject) => {
            const transaction = this.database.transaction([this.storeName], 'readonly');
            const store = transaction.objectStore(this.storeName);
            const request = store.getAllKeys();
            
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }
    
    /**
     * 获取文件元数据
     * @param {string} path - 文件路径
     * @returns {Promise<FileMetadata>}
     */
    async getMetadata(path) {
        return new Promise((resolve, reject) => {
            const transaction = this.database.transaction([this.storeName], 'readonly');
            const store = transaction.objectStore(this.storeName);
            const request = store.get(path);
            
            request.onsuccess = () => {
                if (request.result) {
                    const { content, ...metadata } = request.result;
                    metadata.size = content.byteLength;
                    resolve(metadata);
                } else {
                    reject(new Error(`File not found: ${path}`));
                }
            };
            
            request.onerror = () => reject(request.error);
        });
    }
}

// ============================================================
// File System Access API 后端实现
// ============================================================

/**
 * File System Access API 后端
 * 用于支持 File System Access API 的现代浏览器
 */
class FileSystemAccessBackend {
    /**
     * @param {Function} logError - 错误日志函数
     */
    constructor(logError) {
        /** @type {Function} */
        this.logError = logError;
        
        /** @type {FileSystemDirectoryHandle|null} */
        this.rootDirectory = null;
    }
    
    /**
     * 初始化 - 请求文件系统访问权限
     * @returns {Promise<void>}
     */
    async initialize() {
        try {
            // 请求持久化存储
            if (navigator.storage && navigator.storage.persist) {
                await navigator.storage.persist();
            }
            
            // 获取 OPFS 根目录
            this.rootDirectory = await navigator.storage.getDirectory();
        } catch (error) {
            this.logError('File System Access initialization failed:', error);
            throw error;
        }
    }
    
    /**
     * 读取文件
     * @param {string} path - 文件路径
     * @returns {Promise<ArrayBuffer>}
     */
    async readFile(path) {
        const fileHandle = await this.rootDirectory.getFileHandle(path);
        const file = await fileHandle.getFile();
        return file.arrayBuffer();
    }
    
    /**
     * 写入文件
     * @param {string} path - 文件路径
     * @param {string} mimeType - MIME 类型
     * @param {ArrayBuffer} content - 文件内容
     * @returns {Promise<void>}
     */
    async writeFile(path, mimeType, content) {
        const fileHandle = await this.rootDirectory.getFileHandle(path, { create: true });
        const writable = await fileHandle.createWritable();
        await writable.write(content);
        await writable.close();
    }
    
    /**
     * 修改文件
     */
    async modifyFile(path, mimeType, content) {
        return this.writeFile(path, mimeType, content);
    }
    
    /**
     * 删除文件
     * @param {string} path - 文件路径
     * @returns {Promise<void>}
     */
    async removeFile(path) {
        await this.rootDirectory.removeEntry(path);
    }
    
    /**
     * 列出所有文件
     * @returns {Promise<string[]>}
     */
    async listFiles() {
        const files = [];
        for await (const [name, handle] of this.rootDirectory.entries()) {
            if (handle.kind === 'file') {
                files.push(name);
            }
        }
        return files;
    }
    
    /**
     * 获取文件元数据
     * @param {string} path - 文件路径
     * @returns {Promise<FileMetadata>}
     */
    async getMetadata(path) {
        const fileHandle = await this.rootDirectory.getFileHandle(path);
        const file = await fileHandle.getFile();
        return {
            path,
            name: file.name,
            size: file.size,
            mimeType: file.type || MIME_TYPE_KMZ,
            lastModified: new Date(file.lastModified).toISOString()
        };
    }
}

// ============================================================
// 主类: LocalFileSystem
// ============================================================

/**
 * Google Earth 本地文件系统
 * 
 * 提供统一的文件系统 API，自动选择最佳后端:
 * - 现代浏览器: File System Access API (OPFS)
 * - 旧版浏览器: IndexedDB
 * 
 * 与 Google Earth Web 主模块通过 window.Module.* 通信
 * 
 * @example
 * // 全局命名空间访问
 * const fs = geo.earth.app.localfilesystem.web.EarthLocalFileSystem;
 * fs.onInitAgent();
 * fs.onReadFile('myfile.kmz');
 */
class LocalFileSystem {
    constructor() {
        /**
         * 文件系统后端
         * @type {IndexedDBBackend|FileSystemAccessBackend}
         */
        this.backend = null;
        
        /**
         * 是否已初始化
         * @type {boolean}
         */
        this.initialized = false;
        
        /**
         * 错误日志函数
         * @type {Function}
         */
        this.logError = console.error.bind(console);
        
        /**
         * 本地存储引用
         * @type {Storage}
         */
        this.localStorage = GLOBAL.localStorage;
        
        /**
         * 任务队列 (确保操作按顺序执行)
         * @type {Promise}
         */
        this.taskQueue = Promise.resolve();
        
        // 选择并初始化后端
        this._selectBackend();
        
        // 如果之前已授权，自动初始化
        if (this.localStorage.getItem(STORAGE_KEY_PERMISSION) === "true") {
            this.initialize();
        }
        
        // 通知主模块本地存储可用
        this._notifyModule('PersistentLocalStorageAvailable', true);
    }
    
    /**
     * 选择文件系统后端
     * @private
     */
    _selectBackend() {
        const hasFileSystemAccess = 
            typeof GLOBAL.showDirectoryPicker === 'function' ||
            (navigator.storage && typeof navigator.storage.getDirectory === 'function');
        
        if (hasFileSystemAccess) {
            this.backend = new FileSystemAccessBackend(this.logError);
        } else {
            this.backend = new IndexedDBBackend(this.logError);
        }
    }
    
    /**
     * 通知主模块
     * @private
     * @param {string} event - 事件名称
     * @param {...any} args - 参数
     */
    _notifyModule(event, ...args) {
        const callback = window.Module?.[`LocalFileSystem_${event}`];
        if (typeof callback === 'function') {
            callback(...args);
        }
    }
    
    // ========================================================
    // 事件处理器 (供主模块调用)
    // ========================================================
    
    /**
     * 初始化代理事件处理
     */
    onInitAgent() {
        this.taskQueue = this.taskQueue.then(() => this.initialize());
    }
    
    /**
     * 添加文件事件处理
     * @param {string} path - 文件路径
     * @param {ArrayBuffer} content - 文件内容
     */
    onAddFile(path, content) {
        this.taskQueue = this.taskQueue.then(() => this.writeFile(path, content));
    }
    
    /**
     * 修改文件事件处理
     * @param {string} path - 文件路径
     * @param {ArrayBuffer} content - 文件内容
     */
    onModifyFile(path, content) {
        this.taskQueue = this.taskQueue.then(() => this.modifyFile(path, content));
    }
    
    /**
     * 删除文件事件处理
     * @param {string} path - 文件路径
     */
    onRemoveFile(path) {
        this.taskQueue = this.taskQueue.then(() => this.removeFile(path));
    }
    
    /**
     * 读取文件事件处理
     * @param {string} path - 文件路径
     */
    onReadFile(path) {
        this.taskQueue = this.taskQueue.then(() => this.readFile(path));
    }
    
    /**
     * 列出文件事件处理
     */
    onListFiles() {
        this.taskQueue = this.taskQueue.then(() => this.listFiles());
    }
    
    /**
     * 加载文件元数据事件处理
     * @param {string} path - 文件路径
     */
    onLoadFileMetadata(path) {
        this.taskQueue = this.taskQueue.then(() => this.getMetadata(path));
    }
    
    // ========================================================
    // 核心 API
    // ========================================================
    
    /**
     * 初始化文件系统
     * @returns {Promise<void>}
     */
    async initialize() {
        // 防止重复初始化
        if (this.initialized) return;
        this.initialized = true;
        
        try {
            await this.backend.initialize();
            
            // 记录权限状态
            this.localStorage.setItem(STORAGE_KEY_PERMISSION, "true");
            
            // 通知主模块初始化成功
            this._notifyModule('InitAgentSuccess');
            
        } catch (error) {
            this.logError('LocalFileSystem initialization failed:', error);
            
            // 记录权限失败
            this.localStorage.setItem(STORAGE_KEY_PERMISSION, "false");
            
            // 通知主模块初始化失败
            this._notifyModule('InitAgentError', error, false);
        }
    }
    
    /**
     * 读取文件
     * @param {string} path - 文件路径
     * @returns {Promise<void>}
     */
    async readFile(path) {
        try {
            const content = await this.backend.readFile(path);
            this._notifyModule('ReadFileSuccess', path, content);
        } catch (error) {
            const message = error.message || String(error);
            this.logError('readFile failed:', message);
            this._notifyModule('ReadFileError', path, message);
        }
    }
    
    /**
     * 写入文件 (添加新文件)
     * @param {string} path - 文件路径
     * @param {ArrayBuffer} content - 文件内容
     * @returns {Promise<void>}
     */
    async writeFile(path, content) {
        try {
            await this.backend.writeFile(path, MIME_TYPE_KMZ, content);
            this._notifyModule('AddFileSuccess', path);
        } catch (error) {
            const message = error.message || String(error);
            this.logError('writeFile failed:', message);
            this._notifyModule('AddFileError', path, message);
        }
    }
    
    /**
     * 修改文件
     * @param {string} path - 文件路径
     * @param {ArrayBuffer} content - 文件内容
     * @returns {Promise<void>}
     */
    async modifyFile(path, content) {
        try {
            await this.backend.modifyFile(path, MIME_TYPE_KMZ, content);
            this._notifyModule('ModifyFileSuccess', path);
        } catch (error) {
            const message = error.message || String(error);
            this.logError('modifyFile failed:', message);
            this._notifyModule('ModifyFileError', path, message);
        }
    }
    
    /**
     * 删除文件
     * @param {string} path - 文件路径
     * @returns {Promise<void>}
     */
    async removeFile(path) {
        try {
            await this.backend.removeFile(path);
            this._notifyModule('RemoveFileSuccess', path);
        } catch (error) {
            const message = error.message || String(error);
            this.logError('removeFile failed:', message);
            this._notifyModule('RemoveFileError', path, message);
        }
    }
    
    /**
     * 列出所有文件
     * @returns {Promise<void>}
     */
    async listFiles() {
        try {
            const files = await this.backend.listFiles();
            this._notifyModule('ListFilesSuccess', files);
        } catch (error) {
            const message = error.message || String(error);
            this.logError('listFiles failed:', message);
            this._notifyModule('ListFilesError', message);
        }
    }
    
    /**
     * 获取文件元数据
     * @param {string} path - 文件路径
     * @returns {Promise<void>}
     */
    async getMetadata(path) {
        try {
            const metadata = await this.backend.getMetadata(path);
            this._notifyModule('LoadFileMetadataSuccess', path, metadata);
        } catch (error) {
            const message = error.message || String(error);
            this.logError('getMetadata failed:', message);
            this._notifyModule('LoadFileMetadataError', path, message);
        }
    }
}

// ============================================================
// 全局命名空间注册
// ============================================================

/**
 * 注册到 Google Earth 全局命名空间
 * geo.earth.app.localfilesystem.web.EarthLocalFileSystem
 */
(function registerGlobalNamespace() {
    const namespaces = ['geo', 'earth', 'app', 'localfilesystem', 'web'];
    let current = GLOBAL;
    
    for (const ns of namespaces) {
        if (!(ns in current)) {
            current[ns] = {};
        }
        current = current[ns];
    }
    
    // 创建并注册实例
    current.EarthLocalFileSystem = new LocalFileSystem();
})();

// 通知主模块脚本加载完成
window.Module?.LocalFileSystem_JsScriptLoaded?.();

// ============================================================
// 导出 (用于模块环境)
// ============================================================

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        LocalFileSystem,
        IndexedDBBackend,
        FileSystemAccessBackend,
        MIME_TYPE_KMZ
    };
}
