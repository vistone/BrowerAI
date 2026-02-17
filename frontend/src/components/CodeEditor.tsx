import React, { useState } from "react";
import "../styles/CodeEditor.css";

interface CodeEditorProps {
  title: string;
  language: "html" | "css";
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  loading?: boolean;
}

export const CodeEditor: React.FC<CodeEditorProps> = ({
  title,
  language,
  value,
  onChange,
  onSubmit,
  loading = false,
}) => {
  const [lineCount, setLineCount] = useState(0);

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const text = e.target.value;
    onChange(text);
    setLineCount(text.split("\n").length);
  };

  return (
    <div className="code-editor">
      <div className="editor-header">
        <h3>{title}</h3>
        <span className="line-count">{lineCount} 行</span>
      </div>
      <div className="editor-container">
        <textarea
          className="code-input"
          value={value}
          onChange={handleChange}
          placeholder={`粘贴您的${language === "html" ? "HTML" : "CSS"}代码...`}
          spellCheck={false}
          disabled={loading}
        />
      </div>
      <div className="editor-footer">
        <button
          className="submit-btn"
          onClick={onSubmit}
          disabled={loading || value.trim().length === 0}
        >
          {loading ? "处理中..." : `提交${language === "html" ? "HTML" : "CSS"}`}
        </button>
        <span className="char-count">{value.length} 字符</span>
      </div>
    </div>
  );
};

export default CodeEditor;
