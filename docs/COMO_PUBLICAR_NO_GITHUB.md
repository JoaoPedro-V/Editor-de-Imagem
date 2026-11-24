# Como publicar o PixelLab no GitHub

## Passo a passo recomendado: GitHub Desktop (Windows)

1. Entre em `https://github.com` e faça login.
2. Acesse `https://github.com/new` e crie um repositório chamado `pixellab-editor-de-imagens`.
3. Escolha **Public** se o professor precisa acessar sem convite. Escolha **Private** se preferir restringir o acesso; nesse caso, adicione o professor como colaborador.
4. Deixe desmarcadas as opções para adicionar README, `.gitignore` e licença. Esses arquivos já estão no pacote do projeto. Clique em **Create repository**.
5. Instale e abra o GitHub Desktop: `https://desktop.github.com/`.
6. No GitHub Desktop, escolha **File → Clone Repository**, selecione o repositório `pixellab-editor-de-imagens` e clone para uma pasta local.
7. Extraia `PixelLab_GitHub_Starter.zip`. Copie todo o conteúdo de dentro da pasta `PixelLab-Editor-de-Imagens` para a pasta local que acabou de clonar. **Não apague nem substitua a pasta oculta `.git`** criada pelo GitHub Desktop. Ative a visualização de itens ocultos do Windows para copiar também `.gitignore` e `.github`.
8. Volte ao GitHub Desktop. Os arquivos do PixelLab deverão aparecer em **Changes**. Na mensagem de resumo, escreva `Versão inicial do PixelLab` e clique em **Commit to main**.
9. Clique em **Push origin** para enviar os arquivos ao GitHub.

## Opção pelo terminal (Git instalado)

Abra o terminal dentro da pasta do repositório clonado, depois de copiar os arquivos do pacote, e execute:

```bash
git add .
git commit -m "Versão inicial do PixelLab"
git push origin main
```

Se estiver criando o repositório local do zero em vez de cloná-lo, será necessário inicializar o Git e configurar o endereço remoto antes do primeiro push.

## O que vai para o repositório

- Notebook para Google Colab.
- Aplicação Python.
- Dependências (`requirements.txt`).
- Testes automatizados.
- README e documentação técnica.
- `.gitignore` para excluir arquivos temporários e ambientes virtuais.
- GitHub Actions para executar os testes a cada push e pull request.

Não envie tokens, senhas, arquivos `.env`, dados pessoais ou imagens privadas. O `.gitignore` já exclui arquivos `.env`, ambientes virtuais e caches Python.
