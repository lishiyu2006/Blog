# vue-router导入方法

router的导入方法不同会决定数据是懒加载还是提前渲染好

显式的导入就不是懒加载入`improt...from...`,而`component:()=>import(.../views/...)`