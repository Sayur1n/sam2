import React from 'react';
import { styled } from '@stylexjs/stylex';
import { colors } from '../theme/colors';
import { OCRProcessor } from '../common/components/ocr/OCRProcessor';

const PageContainer = styled('div', {
  display: 'flex',
  flexDirection: 'column',
  height: '100vh',
  backgroundColor: colors.background,
  color: colors.text,
});

const Header = styled('div', {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: '16px 24px',
  borderBottom: `1px solid ${colors.border}`,
  backgroundColor: colors.surface,
});

const Title = styled('h1', {
  margin: 0,
  fontSize: '24px',
  fontWeight: '600',
  color: colors.text,
});

const Description = styled('p', {
  margin: '8px 0 0 0',
  fontSize: '14px',
  color: colors.textSecondary,
});

const Content = styled('div', {
  flex: 1,
  overflow: 'hidden',
});

export default function OCRPage() {
  return (
    <PageContainer>
      <Header>
        <div>
          <Title>OCR文字替换工具</Title>
          <Description>
            上传图像，自动识别文字并替换为翻译内容，支持文字图层编辑
          </Description>
        </div>
      </Header>
      <Content>
        <OCRProcessor />
      </Content>
    </PageContainer>
  );
} 