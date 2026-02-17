// 真实的API客户端 - 与后端BrowerAI API通信

const API_BASE = import.meta.env.VITE_API_URL || "/api";

export interface RenderRequest {
  html: string;
  css?: string;
  use_ai?: boolean;
}

export interface RenderResponse {
  success: boolean;
  message: string;
  rules_count: number;
  ai_enhanced?: boolean;
  duration_ms?: number;
}

export interface ParseCssRequest {
  css: string;
  use_ai?: boolean;
}

export interface ParseCssResponse {
  success: boolean;
  rules_count: number;
  ai_enhanced?: boolean;
  predicted_properties?: Array<{ name: string; confidence: number }>;
  duration_ms?: number;
}

export interface ParseHtmlRequest {
  html: string;
}

export interface ParseHtmlResponse {
  success: boolean;
  node_count: number;
  depth: number;
  message?: string;
  duration_ms?: number;
}

export interface HealthResponse {
  status: "ok" | "degraded" | "unhealthy" | "error";
  version: string;
  ai_enabled: boolean;
}

export interface VersionResponse {
  version: string;
  phase: string;
  features: string[];
}

/**
 * 真实的API客户端 - 与BrowerAI后端通信
 */
class BrowerAIClient {
  private baseUrl: string;
  private timeout: number = 30000; // 30秒超时

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  /**
   * 健康检查
   */
  async health(): Promise<HealthResponse> {
    return this.get<HealthResponse>("/health");
  }

  /**
   * 获取版本信息
   */
  async version(): Promise<VersionResponse> {
    return this.get<VersionResponse>("/version");
  }

  /**
   * 渲染HTML/CSS
   */
  async render(request: RenderRequest): Promise<RenderResponse> {
    const startTime = performance.now();
    try {
      const response = await this.post<RenderResponse>("/v1/render", request);
      const duration = performance.now() - startTime;
      return { ...response, duration_ms: Math.round(duration) };
    } catch (error) {
      console.error("渲染失败:", error);
      throw error;
    }
  }

  /**
   * 解析CSS
   */
  async parseCss(request: ParseCssRequest): Promise<ParseCssResponse> {
    const startTime = performance.now();
    try {
      const response = await this.post<ParseCssResponse>("/v1/parse/css", request);
      const duration = performance.now() - startTime;
      return { ...response, duration_ms: Math.round(duration) };
    } catch (error) {
      console.error("CSS解析失败:", error);
      throw error;
    }
  }

  /**
   * 解析HTML
   */
  async parseHtml(request: ParseHtmlRequest): Promise<ParseHtmlResponse> {
    const startTime = performance.now();
    try {
      const response = await this.post<ParseHtmlResponse>("/v1/parse/html", request);
      const duration = performance.now() - startTime;
      return { ...response, duration_ms: Math.round(duration) };
    } catch (error) {
      console.error("HTML解析失败:", error);
      throw error;
    }
  }

  /**
   * 执行GET请求
   */
  private async get<T>(endpoint: string): Promise<T> {
    const response = await this.fetchWithTimeout(`${this.baseUrl}${endpoint}`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });
    return response.json();
  }

  /**
   * 执行POST请求
   */
  private async post<T>(endpoint: string, body: unknown): Promise<T> {
    const response = await this.fetchWithTimeout(`${this.baseUrl}${endpoint}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });
    return response.json();
  }

  /**
   * 带超时的fetch
   */
  private fetchWithTimeout(url: string, options: RequestInit): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    return fetch(url, { ...options, signal: controller.signal })
      .then(async (response) => {
        clearTimeout(timeoutId);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return response;
      })
      .catch((error) => {
        clearTimeout(timeoutId);
        throw error;
      });
  }
}

export const apiClient = new BrowerAIClient();
export default BrowerAIClient;
