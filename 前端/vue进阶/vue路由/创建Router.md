# 创建Router

在index.js中定义路由使用createRouter函数

~~~js
createRouter({  
  history: createWebHistory(import.meta.env.BASE_URL),  
  routes: [  
    {      path:"/",  
      name:"home",  
      component:HomeView  
    },  
    {  
      path:"/about",  
      name:"about",  
      component:AboutView  
    }  
  ],  
})
~~~

并将他赋值给router并传出给main.js在main.js里app.use(router),但是这一行的操作在创建vue示例的时候选择了router选项自动包含,上述只是解释含义
