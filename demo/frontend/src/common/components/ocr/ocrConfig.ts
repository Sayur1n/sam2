// OCR配置文件
export const OCR_CONFIG = {
  // OCR服务地址 - 可以配置为独立服务或集成服务
  API_BASE_URL: process.env.NODE_ENV === 'development' 
    ? 'http://localhost:5000'  // 开发环境使用独立OCR服务
    : '',  // 生产环境使用相对路径（集成到主服务）
  
  // API端点
  ENDPOINTS: {
    DETECT: '/api/ocr/detect',
    PROCESS: '/api/ocr/process', 
    GENERATE: '/api/ocr/generate',
    TRANSLATE: '/api/ocr/translate',
  },
  
  // 获取完整的API URL
  getApiUrl: (endpoint: string) => {
    return OCR_CONFIG.API_BASE_URL + endpoint;
  }
};

// 导出默认配置
export default OCR_CONFIG; 