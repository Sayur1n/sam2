import React, { useState, useRef, useCallback, useEffect } from 'react';
import { styled } from '@stylexjs/stylex';
import { colors } from '../../../theme/colors';
import OCR_CONFIG from './ocrConfig';

interface TextLayer {
  id: string;
  original_text: string;
  translated_text: string;
  box: [number, number, number, number]; // [x1, y1, x2, y2]
  text_color: [number, number, number];
  font_size: number;
  visible: boolean;
}

interface OCREditorProps {
  imageUrl: string;
  initialLayers?: TextLayer[];
  onLayersChange: (layers: TextLayer[]) => void;
  onGenerate: (finalImage: string) => void;
}

const EditorContainer = styled('div', {
  display: 'flex',
  flexDirection: 'column',
  height: '100%',
  backgroundColor: colors.background,
  color: colors.text,
});

const ImageContainer = styled('div', {
  position: 'relative',
  flex: 1,
  overflow: 'hidden',
  border: `1px solid ${colors.border}`,
  borderRadius: '8px',
  margin: '8px',
});

const Canvas = styled('canvas', {
  display: 'block',
  maxWidth: '100%',
  maxHeight: '100%',
});

const ControlsPanel = styled('div', {
  display: 'flex',
  flexDirection: 'column',
  gap: '12px',
  padding: '16px',
  borderTop: `1px solid ${colors.border}`,
  backgroundColor: colors.surface,
});

const Button = styled('button', {
  padding: '8px 16px',
  borderRadius: '6px',
  border: 'none',
  cursor: 'pointer',
  fontSize: '14px',
  fontWeight: '500',
  transition: 'all 0.2s ease',
  ':hover': {
    opacity: 0.8,
  },
});

const PrimaryButton = styled(Button, {
  backgroundColor: colors.primary,
  color: colors.onPrimary,
});

const SecondaryButton = styled(Button, {
  backgroundColor: colors.surface,
  color: colors.text,
  border: `1px solid ${colors.border}`,
});

const LayersPanel = styled('div', {
  maxHeight: '200px',
  overflowY: 'auto',
  border: `1px solid ${colors.border}`,
  borderRadius: '6px',
  padding: '8px',
});

const LayerItem = styled('div', {
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
  padding: '8px',
  borderBottom: `1px solid ${colors.border}`,
  cursor: 'pointer',
  ':hover': {
    backgroundColor: colors.surfaceHover,
  },
});

const LayerText = styled('div', {
  flex: 1,
  fontSize: '12px',
});

const ColorPicker = styled('input', {
  width: '30px',
  height: '30px',
  border: 'none',
  borderRadius: '4px',
  cursor: 'pointer',
});

const FontSizeInput = styled('input', {
  width: '60px',
  padding: '4px 8px',
  border: `1px solid ${colors.border}`,
  borderRadius: '4px',
  fontSize: '12px',
});

const VisibilityToggle = styled('input', {
  cursor: 'pointer',
});

export const OCREditor: React.FC<OCREditorProps> = ({
  imageUrl,
  initialLayers = [],
  onLayersChange,
  onGenerate,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [layers, setLayers] = useState<TextLayer[]>([]);
  const [selectedLayer, setSelectedLayer] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageElement, setImageElement] = useState<HTMLImageElement | null>(null);

  // 从props接收初始图层数据
  useEffect(() => {
    if (initialLayers.length > 0) {
      setLayers(initialLayers);
    }
  }, [initialLayers]);

  // 加载图像
  useEffect(() => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      setImageElement(img);
      setImageLoaded(true);
      drawCanvas();
    };
    img.src = imageUrl;
  }, [imageUrl]);

  // 绘制画布
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx || !imageElement) return;

    // 设置画布尺寸
    canvas.width = imageElement.width;
    canvas.height = imageElement.height;

    // 绘制背景图像
    ctx.drawImage(imageElement, 0, 0);

    // 绘制文字图层
    layers.forEach((layer) => {
      if (!layer.visible) return;

      const [x1, y1, x2, y2] = layer.box;
      const centerX = (x1 + x2) / 2;
      const centerY = (y1 + y2) / 2;

      ctx.fillStyle = `rgb(${layer.text_color.join(',')})`;
      ctx.font = `${layer.font_size}px Arial`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      // 绘制文字
      ctx.fillText(layer.translated_text, centerX, centerY);

      // 如果是选中的图层，绘制边框
      if (layer.id === selectedLayer) {
        ctx.strokeStyle = '#007AFF';
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      }
    });
  }, [imageElement, layers, selectedLayer]);

  useEffect(() => {
    if (imageLoaded) {
      drawCanvas();
    }
  }, [imageLoaded, layers, selectedLayer, drawCanvas]);

  // 处理鼠标事件
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);

    // 检查是否点击了图层
    const clickedLayer = layers.find((layer) => {
      const [x1, y1, x2, y2] = layer.box;
      return x >= x1 && x <= x2 && y >= y1 && y <= y2;
    });

    if (clickedLayer) {
      setSelectedLayer(clickedLayer.id);
      setIsDragging(true);
      setDragStart({ x, y });
    } else {
      setSelectedLayer(null);
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging || !dragStart || !selectedLayer) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);

    const deltaX = x - dragStart.x;
    const deltaY = y - dragStart.y;

    // 更新图层位置
    setLayers((prevLayers) =>
      prevLayers.map((layer) =>
        layer.id === selectedLayer
          ? {
              ...layer,
              box: [
                layer.box[0] + deltaX,
                layer.box[1] + deltaY,
                layer.box[2] + deltaX,
                layer.box[3] + deltaY,
              ] as [number, number, number, number],
            }
          : layer
      )
    );

    setDragStart({ x, y });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    setDragStart(null);
  };

  // 处理图层属性更改
  const updateLayer = (layerId: string, updates: Partial<TextLayer>) => {
    setLayers((prevLayers) =>
      prevLayers.map((layer) =>
        layer.id === layerId ? { ...layer, ...updates } : layer
      )
    );
  };

  // 处理文字颜色更改
  const handleColorChange = (layerId: string, color: string) => {
    const rgb = color.match(/\d+/g)?.map(Number) || [0, 0, 0];
    updateLayer(layerId, { text_color: rgb as [number, number, number] });
  };

  // 处理字体大小更改
  const handleFontSizeChange = (layerId: string, size: string) => {
    const fontSize = parseInt(size) || 20;
    updateLayer(layerId, { font_size: fontSize });
  };

  // 处理可见性切换
  const handleVisibilityToggle = (layerId: string) => {
    setLayers((prevLayers) =>
      prevLayers.map((layer) =>
        layer.id === layerId ? { ...layer, visible: !layer.visible } : layer
      )
    );
  };

  // 处理翻译文字更改
  const handleTextChange = (layerId: string, text: string) => {
    updateLayer(layerId, { translated_text: text });
  };

  // 处理图层大小调整
  const handleResize = (layerId: string, direction: string, delta: number) => {
    setLayers((prevLayers) =>
      prevLayers.map((layer) => {
        if (layer.id !== layerId) return layer;

        const [x1, y1, x2, y2] = layer.box;
        let newBox = [...layer.box] as [number, number, number, number];

        switch (direction) {
          case 'width':
            newBox[2] = x2 + delta;
            break;
          case 'height':
            newBox[3] = y2 + delta;
            break;
        }

        return { ...layer, box: newBox };
      })
    );
  };

  // 生成最终图像
  const handleGenerate = async () => {
    try {
      const response = await fetch(OCR_CONFIG.getApiUrl(OCR_CONFIG.ENDPOINTS.GENERATE), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageUrl,
          text_layers: layers,
        }),
      });

      const data = await response.json();
      if (data.success) {
        onGenerate(data.final_image);
      } else {
        console.error('生成失败:', data.error);
      }
    } catch (error) {
      console.error('生成请求失败:', error);
    }
  };

  // 通知父组件图层变化
  useEffect(() => {
    onLayersChange(layers);
  }, [layers, onLayersChange]);

  return (
    <EditorContainer>
      <ImageContainer>
        <Canvas
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        />
      </ImageContainer>

      <ControlsPanel>
        <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
          <PrimaryButton onClick={handleGenerate}>
            生成最终图像
          </PrimaryButton>
          <SecondaryButton onClick={() => setSelectedLayer(null)}>
            取消选择
          </SecondaryButton>
        </div>

        <LayersPanel>
          <h4 style={{ margin: '0 0 8px 0', fontSize: '14px' }}>文字图层</h4>
          {layers.map((layer) => (
            <LayerItem key={layer.id}>
              <VisibilityToggle
                type="checkbox"
                checked={layer.visible}
                onChange={() => handleVisibilityToggle(layer.id)}
              />
              
              <LayerText>
                <div style={{ fontSize: '10px', color: colors.textSecondary }}>
                  {layer.original_text}
                </div>
                <input
                  type="text"
                  value={layer.translated_text}
                  onChange={(e) => handleTextChange(layer.id, e.target.value)}
                  style={{
                    width: '100%',
                    fontSize: '12px',
                    padding: '2px 4px',
                    border: `1px solid ${colors.border}`,
                    borderRadius: '2px',
                  }}
                />
              </LayerText>

              <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <ColorPicker
                  type="color"
                  value={`rgb(${layer.text_color.join(',')}`}
                  onChange={(e) => handleColorChange(layer.id, e.target.value)}
                />
                <FontSizeInput
                  type="number"
                  value={layer.font_size}
                  onChange={(e) => handleFontSizeChange(layer.id, e.target.value)}
                  min="8"
                  max="100"
                />
              </div>
            </LayerItem>
          ))}
        </LayersPanel>
      </ControlsPanel>
    </EditorContainer>
  );
}; 