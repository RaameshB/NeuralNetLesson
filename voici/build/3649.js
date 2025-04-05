"use strict";(self.webpackChunk_JUPYTERLAB_CORE_OUTPUT=self.webpackChunk_JUPYTERLAB_CORE_OUTPUT||[]).push([[3649],{43649:(t,i,e)=>{e.r(i),e.d(i,{VirtualDOM:()=>n,VirtualElement:()=>s,VirtualElementPass:()=>a,VirtualText:()=>r,h:()=>b,hpass:()=>l});var n,d,o=e(45945);class r{constructor(t){this.type="text",this.content=t}}class s{constructor(t,i,e,n){this.type="element",this.tag=t,this.attrs=i,this.children=e,this.renderer=n}}class a extends s{constructor(t,i,e){super(t,i,[],e||void 0)}}function b(t){let i,e={},n=[];for(let t=1,o=arguments.length;t<o;++t){let o=arguments[t];"string"==typeof o?n.push(new r(o)):o instanceof r||o instanceof s?n.push(o):o instanceof Array?d(n,o):1!==t&&2!==t||!o||"object"!=typeof o||("render"in o?i=o:e=o)}return new s(t,e,n,i);function d(t,i){for(let e of i)"string"==typeof e?t.push(new r(e)):(e instanceof r||e instanceof s)&&t.push(e)}}function l(t){let i={},e=null;if(2===arguments.length){const t=arguments[1];"render"in t?e=t:i=t}else if(3===arguments.length)i=arguments[1],e=arguments[2];else if(arguments.length>3)throw new Error("hpass() should be called with 1, 2, or 3 arguments");return new a(t,i,e)}!function(t){t.a=t.bind(void 0,"a"),t.abbr=t.bind(void 0,"abbr"),t.address=t.bind(void 0,"address"),t.area=t.bind(void 0,"area"),t.article=t.bind(void 0,"article"),t.aside=t.bind(void 0,"aside"),t.audio=t.bind(void 0,"audio"),t.b=t.bind(void 0,"b"),t.bdi=t.bind(void 0,"bdi"),t.bdo=t.bind(void 0,"bdo"),t.blockquote=t.bind(void 0,"blockquote"),t.br=t.bind(void 0,"br"),t.button=t.bind(void 0,"button"),t.canvas=t.bind(void 0,"canvas"),t.caption=t.bind(void 0,"caption"),t.cite=t.bind(void 0,"cite"),t.code=t.bind(void 0,"code"),t.col=t.bind(void 0,"col"),t.colgroup=t.bind(void 0,"colgroup"),t.data=t.bind(void 0,"data"),t.datalist=t.bind(void 0,"datalist"),t.dd=t.bind(void 0,"dd"),t.del=t.bind(void 0,"del"),t.dfn=t.bind(void 0,"dfn"),t.div=t.bind(void 0,"div"),t.dl=t.bind(void 0,"dl"),t.dt=t.bind(void 0,"dt"),t.em=t.bind(void 0,"em"),t.embed=t.bind(void 0,"embed"),t.fieldset=t.bind(void 0,"fieldset"),t.figcaption=t.bind(void 0,"figcaption"),t.figure=t.bind(void 0,"figure"),t.footer=t.bind(void 0,"footer"),t.form=t.bind(void 0,"form"),t.h1=t.bind(void 0,"h1"),t.h2=t.bind(void 0,"h2"),t.h3=t.bind(void 0,"h3"),t.h4=t.bind(void 0,"h4"),t.h5=t.bind(void 0,"h5"),t.h6=t.bind(void 0,"h6"),t.header=t.bind(void 0,"header"),t.hr=t.bind(void 0,"hr"),t.i=t.bind(void 0,"i"),t.iframe=t.bind(void 0,"iframe"),t.img=t.bind(void 0,"img"),t.input=t.bind(void 0,"input"),t.ins=t.bind(void 0,"ins"),t.kbd=t.bind(void 0,"kbd"),t.label=t.bind(void 0,"label"),t.legend=t.bind(void 0,"legend"),t.li=t.bind(void 0,"li"),t.main=t.bind(void 0,"main"),t.map=t.bind(void 0,"map"),t.mark=t.bind(void 0,"mark"),t.meter=t.bind(void 0,"meter"),t.nav=t.bind(void 0,"nav"),t.noscript=t.bind(void 0,"noscript"),t.object=t.bind(void 0,"object"),t.ol=t.bind(void 0,"ol"),t.optgroup=t.bind(void 0,"optgroup"),t.option=t.bind(void 0,"option"),t.output=t.bind(void 0,"output"),t.p=t.bind(void 0,"p"),t.param=t.bind(void 0,"param"),t.pre=t.bind(void 0,"pre"),t.progress=t.bind(void 0,"progress"),t.q=t.bind(void 0,"q"),t.rp=t.bind(void 0,"rp"),t.rt=t.bind(void 0,"rt"),t.ruby=t.bind(void 0,"ruby"),t.s=t.bind(void 0,"s"),t.samp=t.bind(void 0,"samp"),t.section=t.bind(void 0,"section"),t.select=t.bind(void 0,"select"),t.small=t.bind(void 0,"small"),t.source=t.bind(void 0,"source"),t.span=t.bind(void 0,"span"),t.strong=t.bind(void 0,"strong"),t.sub=t.bind(void 0,"sub"),t.summary=t.bind(void 0,"summary"),t.sup=t.bind(void 0,"sup"),t.table=t.bind(void 0,"table"),t.tbody=t.bind(void 0,"tbody"),t.td=t.bind(void 0,"td"),t.textarea=t.bind(void 0,"textarea"),t.tfoot=t.bind(void 0,"tfoot"),t.th=t.bind(void 0,"th"),t.thead=t.bind(void 0,"thead"),t.time=t.bind(void 0,"time"),t.title=t.bind(void 0,"title"),t.tr=t.bind(void 0,"tr"),t.track=t.bind(void 0,"track"),t.u=t.bind(void 0,"u"),t.ul=t.bind(void 0,"ul"),t.var_=t.bind(void 0,"var"),t.video=t.bind(void 0,"video"),t.wbr=t.bind(void 0,"wbr")}(b||(b={})),function(t){t.realize=function(t){return d.createDOMNode(t)},t.render=function(t,i){let e=d.hostMap.get(i)||[],n=d.asContentArray(t);d.hostMap.set(i,n),d.updateContent(i,e,n)}}(n||(n={})),function(t){function i(t){let e=arguments[1]||null;const d=arguments[2]||null;if(e)e.insertBefore(i(t),d);else{if("text"===t.type)return document.createTextNode(t.content);if(e=document.createElement(t.tag),function(t,i){for(let e in i)e in n||("on"===e.substr(0,2)?t[e]=i[e]:t.setAttribute(e,i[e]));void 0!==i.className&&t.setAttribute("class",i.className),void 0!==i.htmlFor&&t.setAttribute("for",i.htmlFor),i.dataset&&function(t,i){for(let e in i)t.setAttribute(`data-${e}`,i[e])}(t,i.dataset),i.style&&function(t,i){let e,n=t.style;for(e in i)n[e]=i[e]}(t,i.style)}(e,t.attrs),t.renderer)return t.renderer.render(e,{attrs:t.attrs,children:t.children}),e;for(let n=0,d=t.children.length;n<d;++n)i(t.children[n],e)}return e}function e(t,i,n,d){for(let o=i.length-1;o>=n;--o){const n=i[o],r=d?t.lastChild:t.childNodes[o];"text"===n.type||(n.renderer&&n.renderer.unrender?n.renderer.unrender(r,{attrs:n.attrs,children:n.children}):e(r,n.children,0,!1)),d&&t.removeChild(r)}}t.hostMap=new WeakMap,t.asContentArray=function(t){return t?t instanceof Array?t:[t]:[]},t.createDOMNode=i,t.updateContent=function t(n,r,s){if(r===s)return;let a=function(t,i){let e=t.firstChild,n=Object.create(null);for(let t of i)"element"===t.type&&t.attrs.key&&(n[t.attrs.key]={vNode:t,element:e}),e=e.nextSibling;return n}(n,r),b=r.slice(),l=n.firstChild,v=s.length;for(let e=0;e<v;++e){if(e>=b.length){i(s[e],n);continue}let r=b[e],v=s[e];if(r===v){l=l.nextSibling;continue}if("text"===r.type&&"text"===v.type){l.textContent!==v.content&&(l.textContent=v.content),l=l.nextSibling;continue}if("text"===r.type||"text"===v.type){o.ArrayExt.insert(b,e,v),i(v,n,l);continue}if(!r.renderer!=!v.renderer){o.ArrayExt.insert(b,e,v),i(v,n,l);continue}let c=v.attrs.key;if(c&&c in a){let t=a[c];t.vNode!==r&&(o.ArrayExt.move(b,b.indexOf(t.vNode,e+1),e),n.insertBefore(t.element,l),r=t.vNode,l=t.element)}if(r===v){l=l.nextSibling;continue}let u=r.attrs.key;u&&u!==c?(o.ArrayExt.insert(b,e,v),i(v,n,l)):r.tag===v.tag?(d(l,r.attrs,v.attrs),v.renderer?v.renderer.render(l,{attrs:v.attrs,children:v.children}):t(l,r.children,v.children),l=l.nextSibling):(o.ArrayExt.insert(b,e,v),i(v,n,l))}e(n,b,v,!0)};const n={key:!0,className:!0,htmlFor:!0,dataset:!0,style:!0};function d(t,i,e){if(i===e)return;let d;for(d in i)d in n||d in e||("on"===d.substr(0,2)?t[d]=null:t.removeAttribute(d));for(d in e)d in n||i[d]===e[d]||("on"===d.substr(0,2)?t[d]=e[d]:t.setAttribute(d,e[d]));i.className!==e.className&&(void 0!==e.className?t.setAttribute("class",e.className):t.removeAttribute("class")),i.htmlFor!==e.htmlFor&&(void 0!==e.htmlFor?t.setAttribute("for",e.htmlFor):t.removeAttribute("for")),i.dataset!==e.dataset&&function(t,i,e){for(let n in i)n in e||t.removeAttribute(`data-${n}`);for(let n in e)i[n]!==e[n]&&t.setAttribute(`data-${n}`,e[n])}(t,i.dataset||{},e.dataset||{}),i.style!==e.style&&function(t,i,e){let n,d=t.style;for(n in i)n in e||(d[n]="");for(n in e)i[n]!==e[n]&&(d[n]=e[n])}(t,i.style||{},e.style||{})}}(d||(d={}))}}]);