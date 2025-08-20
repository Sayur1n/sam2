import React, { useState, useRef } from 'react';
import { styled } from '@stylexjs/stylex';
import { colors } from '../../../theme/colors';
import { OCREditor } from './OCREditor';
import OCR_CONFIG from './ocrConfig';

interface TextLayer {
  id: string;
  original_text: string;
  translated_text: string;
  box: [number, number, number, number];
  text_color: [number, number, number];
  font_size: number;
  visible: boolean;
}

const Container = styled('div', {
  display: 'flex',
  flexDirection: 'column',
  height: '100%',
  backgroundColor: colors.background,
  color: colors.text,
});

const UploadArea = styled('div', {
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  height: '200px',
  border: `2px dashed ${colors.border}`,
  borderRadius: '8px',
  margin: '16px',
  cursor: 'pointer',
  transition: 'all 0.2s ease',
  ':hover': {
    borderColor: colors.primary,
    backgroundColor: colors.surfaceHover,
  },
});

const UploadText = styled('div', {
  fontSize: '16px',
  color: colors.textSecondary,
  marginBottom: '8px',
});

const UploadButton = styled('button', {
  padding: '8px 16px',
  backgroundColor: colors.primary,
  color: colors.onPrimary,
  border: 'none',
  borderRadius: '6px',
  cursor: 'pointer',
  fontSize: '14px',
  fontWeight: '500',
  transition: 'all 0.2s ease',
  ':hover': {
    opacity: 0.8,
  },
});

const ProcessingOverlay = styled('div', {
  position: 'absolute',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  backgroundColor: 'rgba(0, 0, 0, 0.7)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: colors.onPrimary,
  fontSize: '16px',
  zIndex: 1000,
});

const ResultContainer = styled('div', {
  display: 'flex',
  flexDirection: 'column',
  height: '100%',
});

const TabContainer = styled('div', {
  display: 'flex',
  borderBottom: `1px solid ${colors.border}`,
  backgroundColor: colors.surface,
});

const Tab = styled('button', {
  padding: '12px 24px',
  backgroundColor: 'transparent',
  border: 'none',
  cursor: 'pointer',
  fontSize: '14px',
  fontWeight: '500',
  color: colors.textSecondary,
  borderBottom: `2px solid transparent`,
  transition: 'all 0.2s ease',
  ':hover': {
    color: colors.text,
  },
});

const ActiveTab = styled(Tab, {
  color: colors.primary,
  borderBottomColor: colors.primary,
});

const TabContent = styled('div', {
  flex: 1,
  overflow: 'hidden',
});

const FinalImageContainer = styled('div', {
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  padding: '16px',
  height: '100%',
  overflow: 'auto',
});

const FinalImage = styled('img', {
  maxWidth: '100%',
  maxHeight: '100%',
  objectFit: 'contain',
  border: `1px solid ${colors.border}`,
  borderRadius: '8px',
});

const DownloadButton = styled('button', {
  marginTop: '16px',
  padding: '8px 16px',
  backgroundColor: colors.primary,
  color: colors.onPrimary,
  border: 'none',
  borderRadius: '6px',
  cursor: 'pointer',
  fontSize: '14px',
  fontWeight: '500',
  transition: 'all 0.2s ease',
  ':hover': {
    opacity: 0.8,
  },
});

export const OCRProcessor: React.FC = () => {
  const [currentImage, setCurrentImage] = useState<string | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [textLayers, setTextLayers] = useState<TextLayer[]>([]);
  const [finalImage, setFinalImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeTab, setActiveTab] = useState<'editor' | 'result'>('editor');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const imageData = e.target?.result as string;
      setCurrentImage(imageData);
      processImage(imageData);
    };
    reader.readAsDataURL(file);
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const processImage = async (imageData: string) => {
    setIsProcessing(true);
    try {
      const response = await fetch(OCR_CONFIG.getApiUrl(OCR_CONFIG.ENDPOINTS.PROCESS), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageData,
          translation_mapping: {
            'Усиленнаяверсия': '加强版',
            'Зкстракт трав': '草本提取物',
            'Без онемения': '无麻木感',
            'Продлевает + питает': '延长+滋养',
            'Безопасно,не вывываетпривыкания': '安全，不会产生依赖',
            'Цена': '价格',
            'CO скидкой': '有折扣',
            'Быстрый': '快速',
            'зффект: продление более 30 минут': '效果：延长超过30分钟',
            'Секрет мужской ВЫНОСЛИВОСТИ': '男性耐力的秘密',
            'Профессиональное средство': '专业产品'
          }
        }),
      });

      const data = await response.json();
      if (data.success) {
        setProcessedImage(data.processed_image);
        setTextLayers(data.text_layers);
      } else {
        console.error('处理失败:', data.error);
      }
    } catch (error) {
      console.error('处理请求失败:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleLayersChange = (layers: TextLayer[]) => {
    setTextLayers(layers);
  };

  const handleGenerate = (finalImageData: string) => {
    setFinalImage(finalImageData);
    setActiveTab('result');
  };

  const handleDownload = () => {
    if (!finalImage) return;

    const link = document.createElement('a');
    link.href = finalImage;
    link.download = 'translated_image.jpg';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
          const imageData = e.target?.result as string;
          setCurrentImage(imageData);
          processImage(imageData);
        };
        reader.readAsDataURL(file);
      }
    }
  };

  if (!currentImage) {
    return (
      <Container>
        <UploadArea
          onClick={handleUploadClick}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        >
          <UploadText>点击上传图像或拖拽图像到此处</UploadText>
          <UploadButton>选择图像</UploadButton>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileUpload}
            style={{ display: 'none' }}
          />
        </UploadArea>
      </Container>
    );
  }

  return (
    <Container>
      {isProcessing && (
        <ProcessingOverlay>
          正在处理图像...
        </ProcessingOverlay>
      )}

      <TabContainer>
        <ActiveTab
          style={activeTab === 'editor' ? {} : { color: colors.textSecondary, borderBottomColor: 'transparent' }}
          onClick={() => setActiveTab('editor')}
        >
          文字编辑器
        </ActiveTab>
        {finalImage && (
          <Tab
            style={activeTab === 'result' ? { color: colors.primary, borderBottomColor: colors.primary } : {}}
            onClick={() => setActiveTab('result')}
          >
            最终结果
          </Tab>
        )}
      </TabContainer>

      <TabContent>
        {activeTab === 'editor' && processedImage && (
          <OCREditor
            imageUrl={processedImage}
            initialLayers={textLayers}
            onLayersChange={handleLayersChange}
            onGenerate={handleGenerate}
          />
        )}
        {activeTab === 'result' && finalImage && (
          <FinalImageContainer>
            <FinalImage src={finalImage} alt="最终结果" />
            <DownloadButton onClick={handleDownload}>
              下载图像
            </DownloadButton>
          </FinalImageContainer>
        )}
      </TabContent>
    </Container>
  );
}; 