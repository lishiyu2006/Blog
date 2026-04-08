# 第一个Vue应用

## 引入：挂载函数 .mount('#app')

函数中间是类似css的标签写法

这个方法是存在于vue里的

~~~html
<div id="app">
	<input v-model="message" />
</div>

<script src="https://unpkg.com/vue@3/dist/vue.global.js"></script>
<script>
  const { createApp } = Vue
  //结构函数结构
  
  const appObj = {
    data() {
      return {
        /*在没有挂载之前，这里只是一个对象中间的方法中的数值，
        在挂载之后，就会变成一个响应式的变量初始值*/
        message: 'Hello Vue!'
      }
    }
  }
  
  createApp(appObj).mount('#app')
</script>
~~~

## 文本插值

~~~vue
<span>Message: {{ msg }}</span>
~~~


## 响应式变量

~~~js
const appObj = {  
  data() {  
    return {  
      inputText: "",  
    };  
  },  
};
~~~