"use strict";(self.webpackChunk_JUPYTERLAB_CORE_OUTPUT=self.webpackChunk_JUPYTERLAB_CORE_OUTPUT||[]).push([[7537],{27537:(e,t,a)=>{a.r(t),a.d(t,{Cassandra:()=>J,MSSQL:()=>M,MariaSQL:()=>G,MySQL:()=>Y,PLSQL:()=>F,PostgreSQL:()=>E,SQLDialect:()=>R,SQLite:()=>K,StandardSQL:()=>D,keywordCompletion:()=>I,keywordCompletionSource:()=>j,schemaCompletion:()=>Z,schemaCompletionSource:()=>L,sql:()=>V});var n=a(86776),r=a(92930),i=a(28366),s=a(58482);const o=20,l=21;function c(e){return e>=65&&e<=90||e>=97&&e<=122||e>=48&&e<=57}function d(e,t,a){for(let n=!1;;){if(e.next<0)return;if(e.next==t&&!n)return void e.advance();n=a&&!n&&92==e.next,e.advance()}}function u(e,t){for(;95==e.next||c(e.next);)null!=t&&(t+=String.fromCharCode(e.next)),e.advance();return t}function m(e,t){for(;48==e.next||49==e.next;)e.advance();t&&e.next==t&&e.advance()}function p(e,t){for(;;){if(46==e.next){if(t)break;t=!0}else if(e.next<48||e.next>57)break;e.advance()}if(69==e.next||101==e.next)for(e.advance(),43!=e.next&&45!=e.next||e.advance();e.next>=48&&e.next<=57;)e.advance()}function f(e){for(;!(e.next<0||10==e.next);)e.advance()}function g(e,t){for(let a=0;a<t.length;a++)if(t.charCodeAt(a)==e)return!0;return!1}const h=" \t\r\n";function _(e,t,a){let n=Object.create(null);n.true=n.false=5,n.null=n.unknown=6;for(let t of e.split(" "))t&&(n[t]=o);for(let e of t.split(" "))e&&(n[e]=l);for(let e of(a||"").split(" "))e&&(n[e]=24);return n}const b="array binary bit boolean char character clob date decimal double float int integer interval large national nchar nclob numeric object precision real smallint time timestamp varchar varying ",v="absolute action add after all allocate alter and any are as asc assertion at authorization before begin between both breadth by call cascade cascaded case cast catalog check close collate collation column commit condition connect connection constraint constraints constructor continue corresponding count create cross cube current current_date current_default_transform_group current_transform_group_for_type current_path current_role current_time current_timestamp current_user cursor cycle data day deallocate declare default deferrable deferred delete depth deref desc describe descriptor deterministic diagnostics disconnect distinct do domain drop dynamic each else elseif end end-exec equals escape except exception exec execute exists exit external fetch first for foreign found from free full function general get global go goto grant group grouping handle having hold hour identity if immediate in indicator initially inner inout input insert intersect into is isolation join key language last lateral leading leave left level like limit local localtime localtimestamp locator loop map match method minute modifies module month names natural nesting new next no none not of old on only open option or order ordinality out outer output overlaps pad parameter partial path prepare preserve primary prior privileges procedure public read reads recursive redo ref references referencing relative release repeat resignal restrict result return returns revoke right role rollback rollup routine row rows savepoint schema scroll search second section select session session_user set sets signal similar size some space specific specifictype sql sqlexception sqlstate sqlwarning start state static system_user table temporary then timezone_hour timezone_minute to trailing transaction translation treat trigger under undo union unique unnest until update usage user using value values view when whenever where while with without work write year zone ",y={backslashEscapes:!1,hashComments:!1,spaceAfterDashes:!1,slashComments:!1,doubleQuotedStrings:!1,doubleDollarQuotedStrings:!1,unquotedBitLiterals:!1,treatBitsAsBytes:!1,charSetCasts:!1,plsqlQuotingMechanism:!1,operatorChars:"*+-%<>!=&|~^/",specialVar:"?",identifierQuotes:'"',caseInsensitiveIdentifiers:!1,words:_(v,b)};function O(e){return new i.Lu((t=>{var a;let{next:n}=t;if(t.advance(),g(n,h)){for(;g(t.next,h);)t.advance();t.acceptToken(36)}else if(36==n&&e.doubleDollarQuotedStrings){let e=u(t,"");36==t.next&&(t.advance(),function(e,t){e:for(;;){if(e.next<0)return console.log("exit at end",e.pos);if(36==e.next){e.advance();for(let a=0;a<t.length;a++){if(e.next!=t.charCodeAt(a))continue e;e.advance()}if(36==e.next)return void e.advance()}else e.advance()}}(t,e),t.acceptToken(3))}else if(39==n||34==n&&e.doubleQuotedStrings)d(t,n,e.backslashEscapes),t.acceptToken(3);else if(35==n&&e.hashComments||47==n&&47==t.next&&e.slashComments)f(t),t.acceptToken(1);else if(45!=n||45!=t.next||e.spaceAfterDashes&&32!=t.peek(1))if(47==n&&42==t.next){t.advance();for(let e=1;;){let a=t.next;if(t.next<0)break;if(t.advance(),42==a&&47==t.next){if(e--,t.advance(),!e)break}else 47==a&&42==t.next&&(e++,t.advance())}t.acceptToken(2)}else if(101!=n&&69!=n||39!=t.next)if(110!=n&&78!=n||39!=t.next||!e.charSetCasts)if(95==n&&e.charSetCasts)for(let a=0;;a++){if(39==t.next&&a>1){t.advance(),d(t,39,e.backslashEscapes),t.acceptToken(3);break}if(!c(t.next))break;t.advance()}else if(e.plsqlQuotingMechanism&&(113==n||81==n)&&39==t.next&&t.peek(1)>0&&!g(t.peek(1),h)){let e=t.peek(1);t.advance(2),function(e,t){let a="[{<(".indexOf(String.fromCharCode(t)),n=a<0?t:"]}>)".charCodeAt(a);for(;;){if(e.next<0)return;if(e.next==n&&39==e.peek(1))return void e.advance(2);e.advance()}}(t,e),t.acceptToken(3)}else if(40==n)t.acceptToken(7);else if(41==n)t.acceptToken(8);else if(123==n)t.acceptToken(9);else if(125==n)t.acceptToken(10);else if(91==n)t.acceptToken(11);else if(93==n)t.acceptToken(12);else if(59==n)t.acceptToken(13);else if(e.unquotedBitLiterals&&48==n&&98==t.next)t.advance(),m(t),t.acceptToken(22);else if(98!=n&&66!=n||39!=t.next&&34!=t.next){if(48==n&&(120==t.next||88==t.next)||(120==n||88==n)&&39==t.next){let e=39==t.next;for(t.advance();(r=t.next)>=48&&r<=57||r>=97&&r<=102||r>=65&&r<=70;)t.advance();e&&39==t.next&&t.advance(),t.acceptToken(4)}else if(46==n&&t.next>=48&&t.next<=57)p(t,!0),t.acceptToken(4);else if(46==n)t.acceptToken(14);else if(n>=48&&n<=57)p(t,!1),t.acceptToken(4);else if(g(n,e.operatorChars)){for(;g(t.next,e.operatorChars);)t.advance();t.acceptToken(15)}else if(g(n,e.specialVar))t.next==n&&t.advance(),function(e){if(39==e.next||34==e.next||96==e.next){let t=e.next;e.advance(),d(e,t,!1)}else u(e)}(t),t.acceptToken(17);else if(g(n,e.identifierQuotes))d(t,n,!1),t.acceptToken(19);else if(58==n||44==n)t.acceptToken(16);else if(c(n)){let r=u(t,String.fromCharCode(n));t.acceptToken(46==t.next||46==t.peek(-r.length-1)?18:null!==(a=e.words[r.toLowerCase()])&&void 0!==a?a:18)}}else{const a=t.next;t.advance(),e.treatBitsAsBytes?(d(t,a,e.backslashEscapes),t.acceptToken(23)):(m(t,a),t.acceptToken(22))}else t.advance(),d(t,39,e.backslashEscapes),t.acceptToken(3);else t.advance(),d(t,39,!0),t.acceptToken(3);else f(t),t.acceptToken(1);var r}))}const k=O(y),x=i.U1.deserialize({version:14,states:"%vQ]QQOOO#wQRO'#DSO$OQQO'#CwO%eQQO'#CxO%lQQO'#CyO%sQQO'#CzOOQQ'#DS'#DSOOQQ'#C}'#C}O'UQRO'#C{OOQQ'#Cv'#CvOOQQ'#C|'#C|Q]QQOOQOQQOOO'`QQO'#DOO(xQRO,59cO)PQQO,59cO)UQQO'#DSOOQQ,59d,59dO)cQQO,59dOOQQ,59e,59eO)jQQO,59eOOQQ,59f,59fO)qQQO,59fOOQQ-E6{-E6{OOQQ,59b,59bOOQQ-E6z-E6zOOQQ,59j,59jOOQQ-E6|-E6|O+VQRO1G.}O+^QQO,59cOOQQ1G/O1G/OOOQQ1G/P1G/POOQQ1G/Q1G/QP+kQQO'#C}O+rQQO1G.}O)PQQO,59cO,PQQO'#Cw",stateData:",[~OtOSPOSQOS~ORUOSUOTUOUUOVROXSOZTO]XO^QO_UO`UOaPObPOcPOdUOeUOfUOgUOhUO~O^]ORvXSvXTvXUvXVvXXvXZvX]vX_vX`vXavXbvXcvXdvXevXfvXgvXhvX~OsvX~P!jOa_Ob_Oc_O~ORUOSUOTUOUUOVROXSOZTO^tO_UO`UOa`Ob`Oc`OdUOeUOfUOgUOhUO~OWaO~P$ZOYcO~P$ZO[eO~P$ZORUOSUOTUOUUOVROXSOZTO^QO_UO`UOaPObPOcPOdUOeUOfUOgUOhUO~O]hOsoX~P%zOajObjOcjO~O^]ORkaSkaTkaUkaVkaXkaZka]ka_ka`kaakabkackadkaekafkagkahka~Oska~P'kO^]O~OWvXYvX[vX~P!jOWnO~P$ZOYoO~P$ZO[pO~P$ZO^]ORkiSkiTkiUkiVkiXkiZki]ki_ki`kiakibkickidkiekifkigkihki~Oski~P)xOWkaYka[ka~P'kO]hO~P$ZOWkiYki[ki~P)xOasObsOcsO~O",goto:"#hwPPPPPPPPPPPPPPPPPPPPPPPPPPx||||!Y!^!d!xPPP#[TYOZeUORSTWZbdfqT[OZQZORiZSWOZQbRQdSQfTZgWbdfqQ^PWk^lmrQl_Qm`RrseVORSTWZbdfq",nodeNames:"⚠ LineComment BlockComment String Number Bool Null ( ) { } [ ] ; . Operator Punctuation SpecialVar Identifier QuotedIdentifier Keyword Type Bits Bytes Builtin Script Statement CompositeIdentifier Parens Braces Brackets Statement",maxTerm:38,nodeProps:[["isolate",-4,1,2,3,19,""]],skippedNodes:[0,1,2],repeatNodeCount:3,tokenData:"RORO",tokenizers:[0,k],topRules:{Script:[0,25]},tokenPrec:0});function w(e){let t=e.cursor().moveTo(e.from,-1);for(;/Comment/.test(t.name);)t.moveTo(t.from,-1);return t.node}function Q(e,t){let a=e.sliceString(t.from,t.to),n=/^([`'"])(.*)\1$/.exec(a);return n?n[2]:a}function C(e){return e&&("Identifier"==e.name||"QuotedIdentifier"==e.name)}function S(e,t){if("CompositeIdentifier"==t.name){let a=[];for(let n=t.firstChild;n;n=n.nextSibling)C(n)&&a.push(Q(e,n));return a}return[Q(e,t)]}function P(e,t){for(let a=[];;){if(!t||"."!=t.name)return a;let n=w(t);if(!C(n))return a;a.unshift(Q(e,n)),t=w(n)}}const q=new Set("where group having order union intersect except all distinct limit offset fetch for".split(" ")),T=/^\w*$/,U=/^[`'"]?\w*[`'"]?$/;function z(e){return e.self&&"string"==typeof e.self.label}class A{constructor(e,t){this.idQuote=e,this.idCaseInsensitive=t,this.list=[],this.children=void 0}child(e){let t=this.children||(this.children=Object.create(null));return t[e]||(e&&!this.list.some((t=>t.label==e))&&this.list.push(X(e,"type",this.idQuote,this.idCaseInsensitive)),t[e]=new A(this.idQuote,this.idCaseInsensitive))}maybeChild(e){return this.children?this.children[e]:null}addCompletion(e){let t=this.list.findIndex((t=>t.label==e.label));t>-1?this.list[t]=e:this.list.push(e)}addCompletions(e){for(let t of e)this.addCompletion("string"==typeof t?X(t,"property",this.idQuote,this.idCaseInsensitive):t)}addNamespace(e){Array.isArray(e)?this.addCompletions(e):z(e)?this.addNamespace(e.children):this.addNamespaceObject(e)}addNamespaceObject(e){for(let t of Object.keys(e)){let a=e[t],n=null,r=t.replace(/\\?\./g,(e=>"."==e?"\0":e)).split("\0"),i=this;z(a)&&(n=a.self,a=a.children);for(let e=0;e<r.length;e++)n&&e==r.length-1&&i.addCompletion(n),i=i.child(r[e].replace(/\\\./g,"."));i.addNamespace(a)}}}function X(e,t,a,n){return new RegExp("^[a-z_][a-z_\\d]*$",n?"i":"").test(e)?{label:e,type:t}:{label:e,type:t,apply:a+e+a}}let B=x.configure({props:[n.Oh.add({Statement:(0,n.mz)()}),n.b_.add({Statement:(e,t)=>({from:Math.min(e.from+100,t.doc.lineAt(e.from).to),to:e.to}),BlockComment:e=>({from:e.from+2,to:e.to-2})}),(0,r.pn)({Keyword:r._A.keyword,Type:r._A.typeName,Builtin:r._A.standard(r._A.name),Bits:r._A.number,Bytes:r._A.string,Bool:r._A.bool,Null:r._A.null,Number:r._A.number,String:r._A.string,Identifier:r._A.name,QuotedIdentifier:r._A.special(r._A.string),SpecialVar:r._A.special(r._A.name),LineComment:r._A.lineComment,BlockComment:r._A.blockComment,Operator:r._A.operator,"Semi Punctuation":r._A.punctuation,"( )":r._A.paren,"{ }":r._A.brace,"[ ]":r._A.squareBracket})]});class R{constructor(e,t,a){this.dialect=e,this.language=t,this.spec=a}get extension(){return this.language.extension}static define(e){let t=function(e,t,a,n){let r={};for(let t in y)r[t]=(e.hasOwnProperty(t)?e:y)[t];return t&&(r.words=_(t,a||"",n)),r}(e,e.keywords,e.types,e.builtin),a=n.bj.define({name:"sql",parser:B.configure({tokenizers:[{from:k,to:O(t)}]}),languageData:{commentTokens:{line:"--",block:{open:"/*",close:"*/"}},closeBrackets:{brackets:["(","[","{","'",'"',"`"]}}});return new R(t,a,e)}}function j(e,t=!1){return function(e,t){let a=Object.keys(e).map((a=>({label:t?a.toUpperCase():a,type:e[a]==l?"type":e[a]==o?"keyword":"variable",boost:-1})));return(0,s.Ar)(["QuotedIdentifier","SpecialVar","String","LineComment","BlockComment","."],(0,s.et)(a))}(e.dialect.words,t)}function I(e,t=!1){return e.language.data.of({autocomplete:j(e,t)})}function L(e){return e.schema?function(e,t,a,r,i,s){var o;let l=(null===(o=null==s?void 0:s.spec.identifierQuotes)||void 0===o?void 0:o[0])||'"',c=new A(l,!!(null==s?void 0:s.spec.caseInsensitiveIdentifiers)),d=i?c.child(i):null;return c.addNamespace(e),t&&(d||c).addCompletions(t),a&&c.addCompletions(a),d&&c.addCompletions(d.list),r&&c.addCompletions((d||c).child(r).list),e=>{let{parents:t,from:a,quoted:i,empty:s,aliases:o}=function(e,t){let a=(0,n.mv)(e).resolveInner(t,-1),r=function(e,t){let a;for(let e=t;!a;e=e.parent){if(!e)return null;"Statement"==e.name&&(a=e)}let n=null;for(let t=a.firstChild,r=!1,i=null;t;t=t.nextSibling){let a="Keyword"==t.name?e.sliceString(t.from,t.to).toLowerCase():null,s=null;if(r)if("as"==a&&i&&C(t.nextSibling))s=Q(e,t.nextSibling);else{if(a&&q.has(a))break;i&&C(t)&&(s=Q(e,t))}else r="from"==a;s&&(n||(n=Object.create(null)),n[s]=S(e,i)),i=/Identifier$/.test(t.name)?t:null}return n}(e.doc,a);return"Identifier"==a.name||"QuotedIdentifier"==a.name||"Keyword"==a.name?{from:a.from,quoted:"QuotedIdentifier"==a.name?e.doc.sliceString(a.from,a.from+1):null,parents:P(e.doc,w(a)),aliases:r}:"."==a.name?{from:t,quoted:null,parents:P(e.doc,a),aliases:r}:{from:t,quoted:null,parents:[],empty:!0,aliases:r}}(e.state,e.pos);if(s&&!e.explicit)return null;o&&1==t.length&&(t=o[t[0]]||t);let l=c;for(let e of t){for(;!l.children||!l.children[e];)if(l==c&&d)l=d;else{if(l!=d||!r)return null;l=l.child(r)}let t=l.maybeChild(e);if(!t)return null;l=t}let u=i&&e.state.sliceDoc(e.pos,e.pos+1)==i,m=l.list;return l==c&&o&&(m=m.concat(Object.keys(o).map((e=>({label:e,type:"constant"}))))),{from:a,to:u?e.pos+1:void 0,options:(p=i,f=m,p?f.map((e=>Object.assign(Object.assign({},e),{label:e.label[0]==p?e.label:p+e.label+p,apply:void 0}))):f),validFor:i?U:T};var p,f}}(e.schema,e.tables,e.schemas,e.defaultTable,e.defaultSchema,e.dialect||D):()=>null}function Z(e){return e.schema?(e.dialect||D).language.data.of({autocomplete:L(e)}):[]}function V(e={}){let t=e.dialect||D;return new n.Yy(t.language,[Z(e),I(t,!!e.upperCaseKeywords)])}const D=R.define({}),E=R.define({charSetCasts:!0,doubleDollarQuotedStrings:!0,operatorChars:"+-*/<>=~!@#%^&|`?",specialVar:"",keywords:v+"a abort abs absent access according ada admin aggregate alias also always analyse analyze array_agg array_max_cardinality asensitive assert assignment asymmetric atomic attach attribute attributes avg backward base64 begin_frame begin_partition bernoulli bit_length blocked bom c cache called cardinality catalog_name ceil ceiling chain char_length character_length character_set_catalog character_set_name character_set_schema characteristics characters checkpoint class class_origin cluster coalesce cobol collation_catalog collation_name collation_schema collect column_name columns command_function command_function_code comment comments committed concurrently condition_number configuration conflict connection_name constant constraint_catalog constraint_name constraint_schema contains content control conversion convert copy corr cost covar_pop covar_samp csv cume_dist current_catalog current_row current_schema cursor_name database datalink datatype datetime_interval_code datetime_interval_precision db debug defaults defined definer degree delimiter delimiters dense_rank depends derived detach detail dictionary disable discard dispatch dlnewcopy dlpreviouscopy dlurlcomplete dlurlcompleteonly dlurlcompletewrite dlurlpath dlurlpathonly dlurlpathwrite dlurlscheme dlurlserver dlvalue document dump dynamic_function dynamic_function_code element elsif empty enable encoding encrypted end_frame end_partition endexec enforced enum errcode error event every exclude excluding exclusive exp explain expression extension extract family file filter final first_value flag floor following force foreach fortran forward frame_row freeze fs functions fusion g generated granted greatest groups handler header hex hierarchy hint id ignore ilike immediately immutable implementation implicit import include including increment indent index indexes info inherit inherits inline insensitive instance instantiable instead integrity intersection invoker isnull k key_member key_type label lag last_value lead leakproof least length library like_regex link listen ln load location lock locked log logged lower m mapping matched materialized max max_cardinality maxvalue member merge message message_length message_octet_length message_text min minvalue mod mode more move multiset mumps name namespace nfc nfd nfkc nfkd nil normalize normalized nothing notice notify notnull nowait nth_value ntile nullable nullif nulls number occurrences_regex octet_length octets off offset oids operator options ordering others over overlay overriding owned owner p parallel parameter_mode parameter_name parameter_ordinal_position parameter_specific_catalog parameter_specific_name parameter_specific_schema parser partition pascal passing passthrough password percent percent_rank percentile_cont percentile_disc perform period permission pg_context pg_datatype_name pg_exception_context pg_exception_detail pg_exception_hint placing plans pli policy portion position position_regex power precedes preceding prepared print_strict_params procedural procedures program publication query quote raise range rank reassign recheck recovery refresh regr_avgx regr_avgy regr_count regr_intercept regr_r2 regr_slope regr_sxx regr_sxy regr_syy reindex rename repeatable replace replica requiring reset respect restart restore result_oid returned_cardinality returned_length returned_octet_length returned_sqlstate returning reverse routine_catalog routine_name routine_schema routines row_count row_number rowtype rule scale schema_name schemas scope scope_catalog scope_name scope_schema security selective self sensitive sequence sequences serializable server server_name setof share show simple skip slice snapshot source specific_name sqlcode sqlerror sqrt stable stacked standalone statement statistics stddev_pop stddev_samp stdin stdout storage strict strip structure style subclass_origin submultiset subscription substring substring_regex succeeds sum symmetric sysid system system_time t table_name tables tablesample tablespace temp template ties token top_level_count transaction_active transactions_committed transactions_rolled_back transform transforms translate translate_regex trigger_catalog trigger_name trigger_schema trim trim_array truncate trusted type types uescape unbounded uncommitted unencrypted unlink unlisten unlogged unnamed untyped upper uri use_column use_variable user_defined_type_catalog user_defined_type_code user_defined_type_name user_defined_type_schema vacuum valid validate validator value_of var_pop var_samp varbinary variable_conflict variadic verbose version versioning views volatile warning whitespace width_bucket window within wrapper xmlagg xmlattributes xmlbinary xmlcast xmlcomment xmlconcat xmldeclaration xmldocument xmlelement xmlexists xmlforest xmliterate xmlnamespaces xmlparse xmlpi xmlquery xmlroot xmlschema xmlserialize xmltable xmltext xmlvalidate yes",types:b+"bigint int8 bigserial serial8 varbit bool box bytea cidr circle precision float8 inet int4 json jsonb line lseg macaddr macaddr8 money numeric pg_lsn point polygon float4 int2 smallserial serial2 serial serial4 text timetz timestamptz tsquery tsvector txid_snapshot uuid xml"}),N="accessible algorithm analyze asensitive authors auto_increment autocommit avg avg_row_length binlog btree cache catalog_name chain change changed checkpoint checksum class_origin client_statistics coalesce code collations columns comment committed completion concurrent consistent contains contributors convert database databases day_hour day_microsecond day_minute day_second delay_key_write delayed delimiter des_key_file dev_pop dev_samp deviance directory disable discard distinctrow div dual dumpfile enable enclosed ends engine engines enum errors escaped even event events every explain extended fast field fields flush force found_rows fulltext grants handler hash high_priority hosts hour_microsecond hour_minute hour_second ignore ignore_server_ids import index index_statistics infile innodb insensitive insert_method install invoker iterate keys kill linear lines list load lock logs low_priority master master_heartbeat_period master_ssl_verify_server_cert masters max max_rows maxvalue message_text middleint migrate min min_rows minute_microsecond minute_second mod mode modify mutex mysql_errno no_write_to_binlog offline offset one online optimize optionally outfile pack_keys parser partition partitions password phase plugin plugins prev processlist profile profiles purge query quick range read_write rebuild recover regexp relaylog remove rename reorganize repair repeatable replace require resume rlike row_format rtree schedule schema_name schemas second_microsecond security sensitive separator serializable server share show slave slow snapshot soname spatial sql_big_result sql_buffer_result sql_cache sql_calc_found_rows sql_no_cache sql_small_result ssl starting starts std stddev stddev_pop stddev_samp storage straight_join subclass_origin sum suspend table_name table_statistics tables tablespace terminated triggers truncate uncommitted uninstall unlock upgrade use use_frm user_resources user_statistics utc_date utc_time utc_timestamp variables views warnings xa xor year_month zerofill",$=b+"bool blob long longblob longtext medium mediumblob mediumint mediumtext tinyblob tinyint tinytext text bigint int1 int2 int3 int4 int8 float4 float8 varbinary varcharacter precision datetime unsigned signed",W="charset clear edit ego help nopager notee nowarning pager print prompt quit rehash source status system tee",Y=R.define({operatorChars:"*+-%<>!=&|^",charSetCasts:!0,doubleQuotedStrings:!0,unquotedBitLiterals:!0,hashComments:!0,spaceAfterDashes:!0,specialVar:"@?",identifierQuotes:"`",keywords:v+"group_concat "+N,types:$,builtin:W}),G=R.define({operatorChars:"*+-%<>!=&|^",charSetCasts:!0,doubleQuotedStrings:!0,unquotedBitLiterals:!0,hashComments:!0,spaceAfterDashes:!0,specialVar:"@?",identifierQuotes:"`",keywords:v+"always generated groupby_concat hard persistent shutdown soft virtual "+N,types:$,builtin:W}),M=R.define({keywords:v+"trigger proc view index for add constraint key primary foreign collate clustered nonclustered declare exec go if use index holdlock nolock nowait paglock pivot readcommitted readcommittedlock readpast readuncommitted repeatableread rowlock serializable snapshot tablock tablockx unpivot updlock with",types:b+"bigint smallint smallmoney tinyint money real text nvarchar ntext varbinary image hierarchyid uniqueidentifier sql_variant xml",builtin:"binary_checksum checksum connectionproperty context_info current_request_id error_line error_message error_number error_procedure error_severity error_state formatmessage get_filestream_transaction_context getansinull host_id host_name isnull isnumeric min_active_rowversion newid newsequentialid rowcount_big xact_state object_id",operatorChars:"*+-%<>!=^&|/",specialVar:"@"}),K=R.define({keywords:v+"abort analyze attach autoincrement conflict database detach exclusive fail glob ignore index indexed instead isnull notnull offset plan pragma query raise regexp reindex rename replace temp vacuum virtual",types:b+"bool blob long longblob longtext medium mediumblob mediumint mediumtext tinyblob tinyint tinytext text bigint int2 int8 unsigned signed real",builtin:"auth backup bail changes clone databases dbinfo dump echo eqp explain fullschema headers help import imposter indexes iotrace lint load log mode nullvalue once print prompt quit restore save scanstats separator shell show stats system tables testcase timeout timer trace vfsinfo vfslist vfsname width",operatorChars:"*+-%<>!=&|/~",identifierQuotes:'`"',specialVar:"@:?$"}),J=R.define({keywords:"add all allow alter and any apply as asc authorize batch begin by clustering columnfamily compact consistency count create custom delete desc distinct drop each_quorum exists filtering from grant if in index insert into key keyspace keyspaces level limit local_one local_quorum modify nan norecursive nosuperuser not of on one order password permission permissions primary quorum rename revoke schema select set storage superuser table three to token truncate ttl two type unlogged update use user users using values where with writetime infinity NaN",types:b+"ascii bigint blob counter frozen inet list map static text timeuuid tuple uuid varint",slashComments:!0}),F=R.define({keywords:v+"abort accept access add all alter and any arraylen as asc assert assign at attributes audit authorization avg base_table begin between binary_integer body by case cast char_base check close cluster clusters colauth column comment commit compress connected constant constraint crash create current currval cursor data_base database dba deallocate debugoff debugon declare default definition delay delete desc digits dispose distinct do drop else elseif elsif enable end entry exception exception_init exchange exclusive exists external fast fetch file for force form from function generic goto grant group having identified if immediate in increment index indexes indicator initial initrans insert interface intersect into is key level library like limited local lock log logging loop master maxextents maxtrans member minextents minus mislabel mode modify multiset new next no noaudit nocompress nologging noparallel not nowait number_base of off offline on online only option or order out package parallel partition pctfree pctincrease pctused pls_integer positive positiven pragma primary prior private privileges procedure public raise range raw rebuild record ref references refresh rename replace resource restrict return returning returns reverse revoke rollback row rowid rowlabel rownum rows run savepoint schema segment select separate set share snapshot some space split sql start statement storage subtype successful synonym tabauth table tables tablespace task terminate then to trigger truncate type union unique unlimited unrecoverable unusable update use using validate value values variable view views when whenever where while with work",builtin:"appinfo arraysize autocommit autoprint autorecovery autotrace blockterminator break btitle cmdsep colsep compatibility compute concat copycommit copytypecheck define echo editfile embedded feedback flagger flush heading headsep instance linesize lno loboffset logsource longchunksize markup native newpage numformat numwidth pagesize pause pno recsep recsepchar repfooter repheader serveroutput shiftinout show showmode spool sqlblanklines sqlcase sqlcode sqlcontinue sqlnumber sqlpluscompatibility sqlprefix sqlprompt sqlterminator suffix tab term termout timing trimout trimspool ttitle underline verify version wrap",types:b+"ascii bfile bfilename bigserial bit blob dec long number nvarchar nvarchar2 serial smallint string text uid varchar2 xml",operatorChars:"*/+-%<>!=~",doubleQuotedStrings:!0,charSetCasts:!0,plsqlQuotingMechanism:!0})}}]);