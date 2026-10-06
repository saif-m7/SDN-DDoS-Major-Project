function PageContainer({ children }) {
  return (
    <main className="min-h-[calc(100vh-76px)] px-8 py-8">
      <div className="mx-auto max-w-[1600px]">
        {children}
      </div>
    </main>
  );
}

export default PageContainer;