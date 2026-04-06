# 第一个Vue应用

## 引入：挂载函数 .mount('#app')

函数中间是类似css的标签写法

这个方法是存在于vue里的

~~~vue
<div id="app">{{ message }}</div>

<script src="https://unpkg.com/vue@3/dist/vue.global.js"></script>
<script>
  const { createApp } = Vue
  #结构函数结构
  
  const appObj = {
    data() {
      return {
        message: 'Hello Vue!'
      }
    }
  }
  
  createApp(appObj).mount('#app')
</script>
~~~