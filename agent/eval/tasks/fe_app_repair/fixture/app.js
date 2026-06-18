const list = document.getElementById('todoList');
const input = document.getElementById('taskInput');
const count = document.getElementById('count');

function addItem() {
  const value = input.value.trim();
  if (!value) return;
  const li = document.createElement('li');
  li.textContent = value;
  list.appendChild(li);
  input.value = '';
  count.textContent = list.children.length;
}

function clearAll() {
  list.innerHTML = '';
  count.textContent = '0';
}

// Câblage des boutons
document.getElementById('addButton').addEventListener('click', addItme);
document.getElementById('clearButton').addEventListener('click', clearAll);
