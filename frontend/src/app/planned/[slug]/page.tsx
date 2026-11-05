import { notFound, redirect } from "next/navigation";
export default async function PlannedPage({params}:{params:Promise<{slug:string}>}) {
  const {slug}=await params;
  if(slug==="pipelines"||slug==="experiments") redirect(`/${slug}`);
  notFound();
}
