export type Column={name:string;type:string;primary_key:boolean;nullable:boolean;references:{table:string;column:string}[];redacted:boolean};
export type StoredTable={name:string;purpose:string;benefit:string;scope:string;rows_in_scope:number|null;columns:Column[];samples:Record<string,unknown>[]};
export type Relationship={source_table:string;source_column:string;target_table:string;target_column:string;kind:"foreign_key"|"logical";why:string};
export type Layout={title:string;objects:string;benefit:string;search:string};
export type StorageOverview={database_kind:string;schema_revision:string;tables:StoredTable[];relationships:Relationship[];sample_limit:number;layers:string[];engine_layouts:Record<string,Layout>};
export type NativeRecord={id:string;metadata:Record<string,unknown>;vector:{dimension:number;dtype:string;norm:number;first_values:number[];bytes_float32:number};linked_chunk:{table:string;id:string;index_id:string;document_id:string;version_id:string;filename:string;pages:number[];text:string}|null};
export type NativeStorage={index_id:string;name:string;store:string;provider:string;model:string;dimension:number;metric:string;namespace:string;stored_vectors:number;layout:Layout;objects:Record<string,unknown>[];native_schema:{name:string;type:string}[];native_indexes:{name:string;definition:string}[];samples:NativeRecord[];note:string};
