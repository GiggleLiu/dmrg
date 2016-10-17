!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!Fortran version block finding for csr type matrix.
!Complexity: sum over p**2, p is the collection of block size.
!Author: Leo
!Data: Oct. 8. 2015
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!Get the specific sub-block in 2-site vmps update
!
!Parameters
!-------------------------
!fl,o1,o2,fr: ndarray(n=3,4,4,3).
!    the decoupled tensor form.
!indices: 2d array.
!    the desired indices for 'down' axis with size - (nl,hndim,hndim,nr).
!
!Return:
!--------------------------
!res: matrix,
subroutine fget_subblock(fl,o1,o2,fr,indices,res,nl,nr,nhl,nhc,nhr,hndim,ndim)
    implicit none
    integer,intent(in) :: nl,nr,ndim,nhl,nhr,nhc,hndim
    integer,intent(in) :: indices(ndim,4)
    complex*16,intent(in) :: fl(nl,nhl,nl),fr(nr,nhr,nr),o1(nhl,hndim,hndim,nhc),o2(nhc,hndim,hndim,nhr)
    complex*16,intent(out) :: res(ndim,ndim)
    integer :: i,j
    complex*16 :: col_temp_fl(nl,nhl),col_temp_fr(nr,nhr),col_temp_o1(nhl,hndim,nhc),col_temp_o2(nhc,hndim,nhr),&
        temp_fl(nhl),temp_fr(nhr),temp_o1(nhl,nhc),temp_o2(nhc,nhr),temp_c1(1,nhc),temp_c2(1,nhr)
    
    !f2py intent(in) :: fl,o1,o2,fr,indices,nl,nr,nhl,nhc,nhr,hndim,ndim
    !f2py intent(out) :: res

    !prepair datas
    res=0
    do j=1,ndim
        !cache datas
        col_temp_fl=fl(:,:,indices(j,1)+1)
        col_temp_o1=o1(:,:,indices(j,2)+1,:)
        col_temp_o2=o2(:,:,indices(j,3)+1,:)
        col_temp_fr=fr(:,:,indices(j,4)+1)
        do i=1,ndim
            !cache datas
            temp_fl=col_temp_fl(indices(i,1)+1,:)
            temp_o1=col_temp_o1(:,indices(i,2)+1,:)
            temp_o2=col_temp_o2(:,indices(i,3)+1,:)
            temp_fr=col_temp_fr(indices(i,4)+1,:)
            !do the contraction
            temp_c1=matmul(reshape(temp_fl,[1,nhl]),temp_o1)
            temp_c2=matmul(temp_c1,temp_o2)
            res(i,j)=sum(temp_c2(1,:)*temp_fr)
        enddo
    enddo
end subroutine fget_subblock
