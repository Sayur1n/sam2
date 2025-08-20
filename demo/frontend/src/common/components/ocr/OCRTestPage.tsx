import React, { useState } from 'react';
import { styled } from '@stylexjs/stylex';
import { colors } from '../../theme/colors';

const TestContainer = styled('div', {
  padding: '20px',
  maxWidth: '800px',
  margin: '0 auto',
});

const TestSection = styled('div', {
  marginBottom: '20px',
  padding: '16px',
  border: `1px solid ${colors.border}`,
  borderRadius: '8px',
  backgroundColor: colors.surface,
});

const TestButton = styled('button', {
  padding: '8px 16px',
  backgroundColor: colors.primary,
  color: colors.onPrimary,
  border: 'none',
  borderRadius: '6px',
  cursor: 'pointer',
  marginRight: '8px',
  marginBottom: '8px',
  ':hover': {
    opacity: 0.8,
  },
});

const TestImage = styled('img', {
  maxWidth: '100%',
  maxHeight: '300px',
  border: `1px solid ${colors.border}`,
  borderRadius: '4px',
  marginTop: '8px',
});

const TestResult = styled('pre', {
  backgroundColor: colors.background,
  padding: '12px',
  borderRadius: '4px',
  overflow: 'auto',
  fontSize: '12px',
  border: `1px solid ${colors.border}`,
});

export const OCRTestPage: React.FC = () => {
  const [testResults, setTestResults] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);

  const testOCRDetection = async () => {
    setIsLoading(true);
    try {
      // 使用一个测试图像
      const testImageUrl = 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k=';
      
      const response = await fetch('/api/ocr/detect', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: testImageUrl,
        }),
      });

      const data = await response.json();
      setTestResults(data);
    } catch (error) {
      setTestResults({ error: error.message });
    } finally {
      setIsLoading(false);
    }
  };

  const testOCRProcess = async () => {
    setIsLoading(true);
    try {
      // 使用一个测试图像
      const testImageUrl = 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k=';
      
      const response = await fetch('/api/ocr/process', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: testImageUrl,
          translation_mapping: {
            'test': '测试',
          },
        }),
      });

      const data = await response.json();
      setTestResults(data);
    } catch (error) {
      setTestResults({ error: error.message });
    } finally {
      setIsLoading(false);
    }
  };

  const testTranslation = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/ocr/translate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: 'Усиленнаяверсия',
          target_lang: 'zh',
        }),
      });

      const data = await response.json();
      setTestResults(data);
    } catch (error) {
      setTestResults({ error: error.message });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <TestContainer>
      <h1>OCR功能测试</h1>
      
      <TestSection>
        <h3>测试OCR文字检测</h3>
        <TestButton onClick={testOCRDetection} disabled={isLoading}>
          {isLoading ? '测试中...' : '测试文字检测'}
        </TestButton>
      </TestSection>

      <TestSection>
        <h3>测试OCR处理</h3>
        <TestButton onClick={testOCRProcess} disabled={isLoading}>
          {isLoading ? '测试中...' : '测试图像处理'}
        </TestButton>
      </TestSection>

      <TestSection>
        <h3>测试翻译功能</h3>
        <TestButton onClick={testTranslation} disabled={isLoading}>
          {isLoading ? '测试中...' : '测试翻译'}
        </TestButton>
      </TestSection>

      {testResults && (
        <TestSection>
          <h3>测试结果</h3>
          <TestResult>
            {JSON.stringify(testResults, null, 2)}
          </TestResult>
        </TestSection>
      )}
    </TestContainer>
  );
}; 