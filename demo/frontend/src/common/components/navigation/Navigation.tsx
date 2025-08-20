import React from 'react';
import { styled } from '@stylexjs/stylex';
import { colors } from '../../theme/colors';
import { Link, useLocation } from 'react-router-dom';

const NavContainer = styled('nav', {
  display: 'flex',
  alignItems: 'center',
  padding: '12px 24px',
  backgroundColor: colors.surface,
  borderBottom: `1px solid ${colors.border}`,
  gap: '24px',
});

const NavLink = styled(Link, {
  textDecoration: 'none',
  color: colors.textSecondary,
  fontSize: '14px',
  fontWeight: '500',
  padding: '8px 12px',
  borderRadius: '6px',
  transition: 'all 0.2s ease',
  ':hover': {
    color: colors.text,
    backgroundColor: colors.surfaceHover,
  },
});

const ActiveNavLink = styled(NavLink, {
  color: colors.primary,
  backgroundColor: colors.primaryLight,
});

const Logo = styled('div', {
  fontSize: '18px',
  fontWeight: '600',
  color: colors.text,
  marginRight: 'auto',
});

export const Navigation: React.FC = () => {
  const location = useLocation();

  const isActive = (path: string) => {
    return location.pathname === path;
  };

  return (
    <NavContainer>
      <Logo>SAM2 Demo</Logo>
      
      <NavLink
        to="/"
        style={isActive('/') ? { color: colors.primary, backgroundColor: colors.primaryLight } : {}}
      >
        视频编辑
      </NavLink>
      
      <NavLink
        to="/ocr"
        style={isActive('/ocr') ? { color: colors.primary, backgroundColor: colors.primaryLight } : {}}
      >
        OCR文字替换
      </NavLink>
    </NavContainer>
  );
}; 