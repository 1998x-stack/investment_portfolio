window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']]
  },
  options: {
    renderActions: {
      addMenu: [0, '', '']
    }
  }
};

document.addEventListener("DOMContentLoaded", function() {
  document.querySelectorAll('h1, h2, h3, h4, h5, h6').forEach(function(header) {
    header.addEventListener('click', function() {
      header.scrollIntoView({ behavior: 'smooth' });
    });
  });

  document.querySelectorAll('p, li').forEach(function(element) {
    element.addEventListener('dblclick', function() {
      // 检查是否已经有输入框，防止重复添加
      if (element.querySelector('.comment-input')) {
        return;
      }

      // 创建输入框容器
      const inputContainer = document.createElement('div');
      inputContainer.className = 'comment-input-container';

      // 创建输入框
      const input = document.createElement('input');
      input.type = 'text';
      input.className = 'comment-input';
      input.placeholder = '输入你的评论并按回车';

      // 处理输入框失去焦点或按下回车键的事件
      input.addEventListener('blur', function() {
        addComment(element, input.value, inputContainer);
      });
      input.addEventListener('keydown', function(event) {
        if (event.key === 'Enter') {
          addComment(element, input.value, inputContainer);
        }
      });

      // 将输入框容器插入到元素的右侧
      inputContainer.appendChild(input);
      element.style.position = 'relative';
      element.appendChild(inputContainer);
      input.focus();
    });
  });

  function addComment(element, comment, inputContainer) {
    if (comment) {
      const commentNode = document.createElement('div');
      commentNode.className = 'comment';
      commentNode.innerText = comment;
      element.appendChild(commentNode);
    }
    // 移除输入框
    if (inputContainer) {
      element.removeChild(inputContainer);
    }
  }

  // 添加保存按钮
  const saveButton = document.createElement('button');
  saveButton.innerText = '保存文章与评论';
  saveButton.className = 'save-button';
  saveButton.addEventListener('click', saveContent);
  document.body.appendChild(saveButton);

  function saveContent() {
    let content = '';

    // 收集所有标题、段落和评论
    document.querySelectorAll('h1, h2, h3, h4, h5, h6, p, li').forEach(function(element) {
      content += element.outerHTML + '\n';
      const comments = element.querySelectorAll('.comment');
      comments.forEach(function(comment) {
        content += comment.outerHTML + '\n';
      });
    });

    // 创建一个 Blob 对象，并生成下载链接
    const blob = new Blob([content], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'article_with_comments.html';
    a.click();
    URL.revokeObjectURL(url);
  }

  MathJax.typesetPromise();
});