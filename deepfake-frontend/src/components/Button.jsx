import React from 'react';

export const Button = ({ 
  children, 
  onClick, 
  variant = 'primary', 
  icon: Icon,
  disabled = false,
  className = '',
  ...props 
}) => {
  const variants = {
    primary: 'bg-purple-600 hover:bg-purple-700 text-white',
    success: 'bg-green-600 hover:bg-green-700 text-white',
    danger: 'bg-red-600 hover:bg-red-700 text-white',
    neutral: 'bg-slate-800 hover:bg-slate-700 text-slate-300',
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`
        flex items-center gap-2 px-6 py-3 rounded-lg transition
        ${variants[variant]}
        ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
        ${className}
      `}
      {...props}
    >
      {Icon && <Icon size={20} />}
      {children}
    </button>
  );
};
